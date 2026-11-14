//! `mnist-cnn` and `cifar-cnn`: train the v0.3 CNN benchmark networks here and in PyTorch
//! from the same initial weights and the same batch order, and print each epoch's loss,
//! test accuracy and time for both.
//!
//! The run's folder (`bench/cnn/<name>-seed<seed>/` under the target folder) gets
//! `init.safetensors`, the initial weights under PyTorch's keys as `save` writes them, and
//! `order.bin`, each epoch's shuffled order of the training rows as little-endian `u32`s,
//! epoch after epoch. `scripts/bench/torch_cnn.py --mode train` reads both, so the two
//! sides start from the same weights and see the same batches; only the dropout masks
//! differ. The results go to `result.json` in the same folder.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use cavitation_core::kernels::{level, threads};
use cavitation_core::{Conv2dGeom, ParamStore, Pool2dGeom, Rng};
use cavitation_datasets::{cifar10, mnist};
use cavitation_nn::{
    Adam, Batch, BatchNorm2d, Batcher, Conv2d, Dataset, Dropout, Flatten, Health, Init, Linear,
    Loss, MaxPool2d, Relu, Sequential, Trainer, accuracy, save,
};

/// One of the two benchmark networks with its pinned schedule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Net {
    /// `"mnist-cnn"` or `"cifar-cnn"`.
    pub name: &'static str,
    /// The data set folder under `data/` and the script's `--net` value.
    pub data: &'static str,
    /// The batch size.
    pub batch: usize,
    /// The default number of epochs.
    pub epochs: usize,
}

/// The MNIST network's schedule: Adam at 1e-3, batch 64, 5 epochs.
pub const MNIST: Net = Net {
    name: "mnist-cnn",
    data: "mnist",
    batch: 64,
    epochs: 5,
};

/// The CIFAR network's schedule: Adam at 1e-3, batch 128, 10 epochs.
pub const CIFAR: Net = Net {
    name: "cifar-cnn",
    data: "cifar",
    batch: 128,
    epochs: 10,
};

/// The MNIST network, LeNet-5's layout, with PyTorch's default initialisation from
/// `Rng::seed(seed)`: `Conv(1 → 6, 5, pad 2) → ReLU → MaxPool(2) → Conv(6 → 16, 5) → ReLU →
/// MaxPool(2) → Flatten → Linear(400 → 120) → ReLU → Linear(120 → 84) → ReLU →
/// Linear(84 → 10)`.
pub fn mnist_network(params: &mut ParamStore, seed: u64) -> Sequential {
    let mut rng = Rng::seed(seed);
    let g1 = Conv2dGeom::with_options(1, 6, (28, 28), (5, 5), (1, 1), (2, 2), (1, 1))
        .expect("a valid convolution");
    let g2 = Conv2dGeom::new(6, 16, (14, 14), (5, 5)).expect("a valid convolution");
    let pool = |c, s| MaxPool2d::new(Pool2dGeom::new(c, (s, s), (2, 2)).expect("a valid pool"));
    Sequential::new()
        .push(Conv2d::new(params, g1, &mut rng))
        .push(Relu)
        .push(pool(6, 28))
        .push(Conv2d::new(params, g2, &mut rng))
        .push(Relu)
        .push(pool(16, 10))
        .push(Flatten)
        .push(Linear::new(params, 400, 120, &mut rng))
        .push(Relu)
        .push(Linear::new(params, 120, 84, &mut rng))
        .push(Relu)
        .push(Linear::new(params, 84, 10, &mut rng))
}

/// The CIFAR-10 network with PyTorch's default initialisation from `Rng::seed(seed)`: four
/// `Conv(3 × 3, pad 1, no bias) → BatchNorm2d → ReLU` blocks (32, 32, 64 and 64 channels)
/// with `MaxPool(2)` after each pair, then `Dropout(0.25) → Flatten → Linear(4096 → 256) →
/// ReLU → Dropout(0.5) → Linear(256 → 10)`. The convolutions have no bias because the batch
/// norm after each cancels it.
pub fn cifar_network(params: &mut ParamStore, seed: u64) -> Sequential {
    let mut rng = Rng::seed(seed);
    let mut net = Sequential::new();
    for (i, o, s) in [(3, 32, 32), (32, 32, 32), (32, 64, 16), (64, 64, 16)] {
        let g = Conv2dGeom::with_options(i, o, (s, s), (3, 3), (1, 1), (1, 1), (1, 1))
            .expect("a valid convolution");
        net = net
            .push(Conv2d::with_init(
                params,
                g,
                Init::TorchDefault,
                None,
                &mut rng,
            ))
            .push(BatchNorm2d::new(params, o, (s, s)))
            .push(Relu);
        // A pool follows the second block of each size.
        if i == o {
            net = net.push(MaxPool2d::new(
                Pool2dGeom::new(o, (s, s), (2, 2)).expect("a valid pool"),
            ));
        }
    }
    net.push(Dropout::new(0.25).expect("a valid rate"))
        .push(Flatten)
        .push(Linear::new(params, 4096, 256, &mut rng))
        .push(Relu)
        .push(Dropout::new(0.5).expect("a valid rate"))
        .push(Linear::new(params, 256, 10, &mut rng))
}

/// The options of `mnist-cnn` and `cifar-cnn`.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    /// `--seed`: the weights', batch order's and dropout's seed (default 0).
    pub seed: u64,
    /// `--epochs`: the number of epochs (default the network's).
    pub epochs: usize,
    /// `--rows`: train on the first this many training rows only (default all).
    pub rows: Option<usize>,
    /// `--python`: the interpreter for the PyTorch side.
    pub python: String,
    /// `--ours-only`: skip the PyTorch side.
    pub ours_only: bool,
}

impl Options {
    /// Parses the arguments after the command for `net`.
    ///
    /// # Errors
    ///
    /// Returns a message for an unknown argument, a missing value or a value that is not
    /// a positive number (`--seed` may be 0).
    pub fn parse(net: Net, args: &[String]) -> Result<Options, String> {
        let mut o = Options {
            seed: 0,
            epochs: net.epochs,
            rows: None,
            python: std::env::var("CAVITATION_BENCH_PYTHON").unwrap_or_else(|_| "python3".into()),
            ours_only: false,
        };
        let mut rest = args.iter();
        let number = |flag: &str, v: Option<&String>, zero_ok: bool| -> Result<usize, String> {
            let v = v.ok_or_else(|| format!("{flag} needs a value"))?;
            v.parse()
                .ok()
                .filter(|&n| zero_ok || n > 0)
                .ok_or_else(|| format!("{flag} must be a positive number, got `{v}`"))
        };
        while let Some(arg) = rest.next() {
            match arg.as_str() {
                "--seed" => o.seed = number("--seed", rest.next(), true)? as u64,
                "--epochs" => o.epochs = number("--epochs", rest.next(), false)?,
                "--rows" => o.rows = Some(number("--rows", rest.next(), false)?),
                "--python" => o.python = rest.next().ok_or("--python needs a value")?.clone(),
                "--ours-only" => o.ours_only = true,
                other => return Err(format!("unknown argument `{other}`")),
            }
        }
        Ok(o)
    }
}

/// One epoch of one side: its mean training loss, test accuracy and training seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Epoch {
    /// The mean loss over the epoch's batches.
    pub loss: f64,
    /// The share of test images classified right, from 0 to 1.
    pub accuracy: f64,
    /// The seconds the epoch's training steps took; the test pass is not timed.
    pub seconds: f64,
}

/// The data folder of `net`, under the repository's `data/`.
fn data_dir(net: Net) -> PathBuf {
    let folder = if net.data == "cifar" {
        "cifar10"
    } else {
        "mnist"
    };
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../data")
        .join(folder)
}

/// Writes `order.bin`: for each of `epochs` epochs, the order a `Batcher` over `rows` rows
/// with shuffle seed `seed` visits them in, as little-endian `u32`s.
fn write_order(
    path: &Path,
    rows: usize,
    batch: usize,
    epochs: usize,
    seed: u64,
) -> Result<(), String> {
    let mut batcher = Batcher::new(rows, batch, Some(seed));
    let mut bytes = Vec::with_capacity(epochs * rows * 4);
    for _ in 0..epochs {
        batcher.epoch();
        for &i in batcher.order() {
            let i = u32::try_from(i).map_err(|_| "more than 2^32 rows".to_string())?;
            bytes.extend(i.to_le_bytes());
        }
    }
    std::fs::write(path, bytes).map_err(|e| format!("could not write {}: {e}", path.display()))
}

/// Trains `model` here, on `train` with `Trainer::step`, Adam at 1e-3 and cross-entropy, in
/// the order [`write_order`] wrote (the same `Batcher`), and measures each epoch.
fn train_ours(
    model: &Sequential,
    params: &mut ParamStore,
    train: &Dataset,
    test: &Dataset,
    net: Net,
    o: &Options,
) -> Result<Vec<Epoch>, String> {
    let mut trainer = Trainer::new(model, Adam::new(1e-3), Loss::CrossEntropy, net.batch);
    trainer.set_health(Health::quiet());
    let mut batcher = Batcher::new(train.len(), net.batch, Some(o.seed));
    let mut batch = Batch::new(train, net.batch);
    let mut epochs = Vec::with_capacity(o.epochs);
    for e in 1..=o.epochs {
        batcher.epoch();
        let (mut sum, mut steps) = (0.0f64, 0usize);
        let start = Instant::now();
        for range in batcher.batches() {
            let view = batcher.gather(train, range, &mut batch);
            sum += f64::from(
                trainer
                    .step(model, params, view)
                    .map_err(|e| e.to_string())?,
            );
            steps += 1;
        }
        let seconds = start.elapsed().as_secs_f64();
        let epoch = Epoch {
            loss: sum / steps as f64,
            accuracy: accuracy(model, params, test, 500),
            seconds,
        };
        println!(
            "  ours  epoch {e:2}: loss {:.4}, test accuracy {:.2}%, {:.1} s",
            epoch.loss,
            epoch.accuracy * 100.0,
            epoch.seconds
        );
        epochs.push(epoch);
    }
    Ok(epochs)
}

/// Runs `torch_cnn.py --mode train` on the run's files and reads its epoch lines, each
/// `epoch <n> <loss> <accuracy> <seconds>`; returns them with the lines it printed about
/// itself (`torch …`).
fn train_torch(
    net: Net,
    o: &Options,
    dir: &Path,
    rows: usize,
) -> Result<(Vec<Epoch>, Vec<String>), String> {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/bench/torch_cnn.py");
    let out = Command::new(&o.python)
        .arg("-I")
        .arg(&script)
        .args(["--mode", "train", "--net", net.data])
        .arg("--data-dir")
        .arg(data_dir(net))
        .arg("--init")
        .arg(dir.join("init.safetensors"))
        .arg("--order")
        .arg(dir.join("order.bin"))
        .args(["--epochs", &o.epochs.to_string()])
        .args(["--batch", &net.batch.to_string()])
        .args(["--rows", &rows.to_string()])
        .args(["--threads", &threads().to_string()])
        .args(["--seed", &o.seed.to_string()])
        .output()
        .map_err(|e| {
            format!(
                "could not run `{}`: {e}; pass the interpreter that has PyTorch with --python \
                 <path> or CAVITATION_BENCH_PYTHON",
                o.python
            )
        })?;
    if !out.status.success() {
        return Err(format!(
            "torch_cnn.py failed ({}):\n{}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let (mut epochs, mut about) = (Vec::new(), Vec::new());
    for line in text.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        match words.as_slice() {
            ["epoch", _, loss, acc, secs] => {
                let num = |w: &str| w.parse::<f64>().map_err(|_| format!("bad line `{line}`"));
                epochs.push(Epoch {
                    loss: num(loss)?,
                    accuracy: num(acc)?,
                    seconds: num(secs)?,
                });
            }
            ["torch", ..] => about.push(line.to_string()),
            _ => {}
        }
    }
    if epochs.len() != o.epochs {
        return Err(format!(
            "torch_cnn.py printed {} epochs, not {}:\n{text}",
            epochs.len(),
            o.epochs
        ));
    }
    for (e, epoch) in epochs.iter().enumerate() {
        println!(
            "  torch epoch {:2}: loss {:.4}, test accuracy {:.2}%, {:.1} s",
            e + 1,
            epoch.loss,
            epoch.accuracy * 100.0,
            epoch.seconds
        );
    }
    Ok((epochs, about))
}

/// The median of the epoch times after the first, whose time includes the warm-up (the
/// first epoch's alone when there is only one).
pub fn steady_seconds(epochs: &[Epoch]) -> f64 {
    let mut t: Vec<f64> = epochs
        .iter()
        .skip(usize::from(epochs.len() > 1))
        .map(|e| e.seconds)
        .collect();
    t.sort_by(f64::total_cmp);
    let n = t.len();
    if n % 2 == 1 {
        t[n / 2]
    } else {
        (t[n / 2 - 1] + t[n / 2]) / 2.0
    }
}

/// The JSON of one side's epochs.
fn epochs_json(epochs: &[Epoch]) -> String {
    let items: Vec<String> = epochs
        .iter()
        .map(|e| {
            format!(
                "{{\"loss\": {:e}, \"accuracy\": {:e}, \"seconds\": {:e}}}",
                e.loss, e.accuracy, e.seconds
            )
        })
        .collect();
    format!("[{}]", items.join(", "))
}

/// Runs `mnist-cnn` or `cifar-cnn` (`net`) with options `o`: writes the run's files under
/// `dir_root`, trains here, then in PyTorch unless `--ours-only`, prints the comparison and
/// writes `result.json`.
///
/// # Errors
///
/// Returns a message for missing data, a file that cannot be written or a failed side.
pub fn run(net: Net, o: &Options, dir_root: &Path) -> Result<(), String> {
    let (mut train, test) = match net.data {
        "cifar" => cifar10::load(&data_dir(net)),
        _ => mnist::load(&data_dir(net)),
    }
    .map_err(|e| e.to_string())?;
    if let Some(rows) = o.rows {
        if rows > train.len() {
            return Err(format!(
                "--rows {rows} is more than the {} training rows",
                train.len()
            ));
        }
        train = train.subset(0..rows);
    }
    let dir = dir_root
        .join("cnn")
        .join(format!("{}-seed{}", net.name, o.seed));
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("could not create {}: {e}", dir.display()))?;
    let dir = dir.canonicalize().unwrap_or(dir);
    let mut params = ParamStore::new();
    let model = match net.data {
        "cifar" => cifar_network(&mut params, o.seed),
        _ => mnist_network(&mut params, o.seed),
    };
    save(dir.join("init.safetensors"), &model, &params).map_err(|e| e.to_string())?;
    write_order(
        &dir.join("order.bin"),
        train.len(),
        net.batch,
        o.epochs,
        o.seed,
    )?;
    println!(
        "{}: seed {}, {} epochs of {} rows at batch {}, Adam lr 1e-3, kernel {}, {} threads; files in {}",
        net.name,
        o.seed,
        o.epochs,
        train.len(),
        net.batch,
        level(),
        threads(),
        dir.display()
    );
    let ours = train_ours(&model, &mut params, &train, &test, net, o)?;
    let rows = train.len();
    drop(train);
    let (theirs, about) = if o.ours_only {
        (Vec::new(), Vec::new())
    } else {
        train_torch(net, o, &dir, rows)?
    };
    for line in &about {
        println!("  {line}");
    }
    let last = |e: &[Epoch]| e.last().map_or(f64::NAN, |e| e.accuracy * 100.0);
    print!(
        "final test accuracy: ours {:.2}%; steady epoch time: ours {:.2} s",
        last(&ours),
        steady_seconds(&ours)
    );
    if theirs.is_empty() {
        println!();
    } else {
        let (a, b) = (steady_seconds(&ours), steady_seconds(&theirs));
        println!(
            ", torch {b:.2} s, ours / torch {:.2}; torch final test accuracy {:.2}%",
            a / b,
            last(&theirs)
        );
    }
    let json = format!(
        "{{\"net\": \"{}\", \"seed\": {}, \"epochs\": {}, \"rows\": {rows}, \"batch\": {}, \
         \"kernel\": \"{}\", \"threads\": {}, \"ours\": {}, \"torch\": {}, \"torch_about\": [{}]}}\n",
        net.name,
        o.seed,
        o.epochs,
        net.batch,
        level(),
        threads(),
        epochs_json(&ours),
        epochs_json(&theirs),
        about
            .iter()
            .map(|l| format!("\"{}\"", l.replace('\\', "\\\\").replace('"', "\\\"")))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let path = dir.join("result.json");
    std::fs::write(&path, json).map_err(|e| format!("could not write {}: {e}", path.display()))?;
    println!("wrote {}", path.display());
    Ok(())
}

/// Tests for the CNN benchmark: the networks' sizes, the options and the steady time.
#[cfg(test)]
mod tests {
    use super::*;

    /// The parameter count of `net`.
    fn count(net: &Sequential, params: &ParamStore) -> usize {
        net.parameter_names()
            .iter()
            .map(|(_, id)| params.value(*id).data().len())
            .sum()
    }

    /// The networks have PyTorch's parameter counts (61,706 for LeNet-5's layout, 1,117,162
    /// for the CIFAR network) and its state-dict keys.
    #[test]
    fn networks_match_pytorch_layouts() {
        let mut p = ParamStore::new();
        let m = mnist_network(&mut p, 0);
        assert_eq!(count(&m, &p), 61_706);
        let mut p = ParamStore::new();
        let c = cifar_network(&mut p, 0);
        assert_eq!(count(&c, &p), 1_117_162);
        let names: Vec<String> = c.state_names().into_iter().map(|n| n.0).collect();
        assert_eq!(
            names[..5],
            [
                "0.weight",
                "1.weight",
                "1.bias",
                "1.running_mean",
                "1.running_var"
            ]
        );
        assert!(names.contains(&"16.weight".to_string()) && names.contains(&"19.bias".to_string()));
    }

    /// Options parse with their defaults, and refuse bad values.
    #[test]
    fn options() {
        let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let o = Options::parse(
            CIFAR,
            &args(&["--seed", "2", "--rows", "100", "--ours-only"]),
        )
        .unwrap();
        assert_eq!(
            (o.seed, o.epochs, o.rows, o.ours_only),
            (2, 10, Some(100), true)
        );
        assert_eq!(Options::parse(MNIST, &[]).unwrap().epochs, 5);
        assert!(Options::parse(MNIST, &args(&["--epochs", "0"])).is_err());
        assert!(Options::parse(MNIST, &args(&["--what"])).is_err());
    }

    /// The steady time is the median after the first epoch.
    #[test]
    fn steady_time() {
        let e = |seconds| Epoch {
            loss: 0.0,
            accuracy: 0.0,
            seconds,
        };
        assert_eq!(steady_seconds(&[e(9.0), e(2.0), e(4.0), e(3.0)]), 3.0);
        assert_eq!(steady_seconds(&[e(9.0)]), 9.0);
    }
}
