//! Saving and loading weights in the safetensors format, compatible with a PyTorch
//! `state_dict` saved with `safetensors.torch.save_file`.

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use cavitation_core::{Error, KeyErrors, ParamId, ParamStore};
use safetensors::tensor::{Dtype, SafeTensorError, SafeTensors, TensorView, serialize_to_file};

use crate::Sequential;

/// Maps a safetensors failure on `path` to the library's error: a failed read or write is
/// [`Error::Io`], anything wrong with the file's contents is [`Error::Format`].
fn convert(path: &Path, err: SafeTensorError) -> Error {
    match err {
        SafeTensorError::IoError(source) => Error::Io {
            path: path.to_path_buf(),
            source,
        },
        other => Error::Format(other.to_string()),
    }
}

/// Returns [`Error::Format`] telling the user to save with safetensors if `bytes` starts like
/// a `torch.save` file: `PK\x03\x04` (the zip archive `torch.save` writes by default) or a
/// pickle stream (the legacy format, which starts with the byte `0x80`). A safetensors file
/// starts with its header length as 8 little-endian bytes and its header, a JSON object, at
/// byte 8; a header of 128 bytes also starts with `0x80`, so a file whose byte 8 is `{` is
/// never taken for a pickle. A file of 8 bytes or fewer is never taken for a pickle (it cannot
/// hold a safetensors header either, so the parser reports it as a format error).
fn check_not_torch_save(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    let zip = bytes.starts_with(b"PK\x03\x04");
    let pickle = bytes.len() > 8 && bytes[0] == 0x80 && bytes[8] != b'{';
    if zip || pickle {
        return Err(Error::Format(format!(
            "{}: this is a torch.save file (a zip archive or a pickle), not safetensors; save \
             the weights with safetensors.torch.save_file(model.state_dict(), path)",
            path.display()
        )));
    }
    Ok(())
}

/// Writes every parameter of `model` to `path` as an F32 safetensors file, each tensor
/// named by [`Sequential::parameter_names`] (`0.weight`, `2.bias`, ...). The names and the
/// `[outputs, inputs]` weight layout are those of a PyTorch `state_dict`, so the file loads
/// with `safetensors.torch.load_file` into the same network built in PyTorch.
///
/// The file is written to a temporary file in the same folder and then renamed over
/// `path`, so a failed save leaves no partial file. On Unix the temporary file is created
/// with mode 0600 and keeps it through the rename, so the saved file can be read only by
/// its owner; widen it with [`std::fs::set_permissions`] if another user must read it.
///
/// # Errors
///
/// Returns [`Error::Io`] if the file cannot be written.
///
/// # Panics
///
/// Panics if `model` was built from a different [`ParamStore`] than `params`.
///
/// ```
/// use cavitation_core::{ParamStore, Rng};
/// use cavitation_nn::{Linear, Relu, Sequential, load, save};
/// let mut params = ParamStore::new();
/// let net = Sequential::new()
///     .push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)))
///     .push(Relu);
/// let path = std::env::temp_dir()
///     .join(format!("cavitation-doc-save-{}.safetensors", std::process::id()));
/// save(&path, &net, &params)?;
/// load(&path, &net, &mut params)?;
/// std::fs::remove_file(&path).ok();
/// # Ok::<(), cavitation_core::Error>(())
/// ```
pub fn save(
    path: impl AsRef<Path>,
    model: &Sequential<f32>,
    params: &ParamStore<f32>,
) -> Result<(), Error> {
    let path = path.as_ref();
    let names = model.parameter_names();
    let buffers: Vec<Vec<u8>> = names
        .iter()
        .map(|(_, id)| {
            params
                .value(*id)
                .data()
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect()
        })
        .collect();
    let mut views = Vec::with_capacity(names.len());
    for ((name, id), bytes) in names.iter().zip(&buffers) {
        let shape = params.value(*id).shape().dims().to_vec();
        let view = TensorView::new(Dtype::F32, shape, bytes).map_err(|e| convert(path, e))?;
        views.push((name.as_str(), view));
    }
    serialize_to_file(views, None, path).map_err(|e| convert(path, e))
}

/// Reads a safetensors file into the parameters of `model`, strictly: every parameter's key
/// must be present with exactly its shape and the `F32` element type, and the file must hold
/// no other key. Nothing is written into `params` unless every check passes, so a failed
/// load leaves the parameters as they were.
///
/// # Errors
///
/// Returns [`Error::Io`] if the file cannot be read, [`Error::Format`] if it is not a valid
/// safetensors file (for example truncated, or a `torch.save` file, which the message says),
/// [`Error::Keys`] for keys that do not fit: the missing, the unexpected, the wrong-shape and
/// the wrong-type keys are all found in one pass and listed in one error (see
/// [`KeyErrors`]), with the file's own first keys when none of the model's keys is there
/// and a hint when a weight looks transposed.
///
/// # Panics
///
/// Panics if `model` was built from a different [`ParamStore`] than `params`.
///
/// ```
/// use cavitation_core::{Error, ParamStore, Rng};
/// use cavitation_nn::{Linear, Relu, Sequential, load, save};
/// let mut params = ParamStore::new();
/// let net = Sequential::new().push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)));
/// assert!(matches!(
///     load("cavitation-doc-no-such-file.safetensors", &net, &mut params),
///     Err(Error::Io { .. })
/// ));
///
/// // Save a model, load the file into a fresh model with other weights, compare.
/// let path = std::env::temp_dir()
///     .join(format!("cavitation-doc-load-{}.safetensors", std::process::id()));
/// save(&path, &net, &params)?;
/// let mut fresh_params = ParamStore::new();
/// let fresh = Sequential::new().push(Linear::new(&mut fresh_params, 2, 3, &mut Rng::seed(9)));
/// assert_ne!(
///     params.value(net.parameter_names()[0].1).data(),
///     fresh_params.value(fresh.parameter_names()[0].1).data()
/// );
/// load(&path, &fresh, &mut fresh_params)?;
/// for ((_, a), (_, b)) in net.parameter_names().into_iter().zip(fresh.parameter_names()) {
///     assert_eq!(params.value(a).data(), fresh_params.value(b).data());
/// }
///
/// // A model whose Linear layer sits at index 1 does not match this file's keys `0.*`:
/// // one error lists every key that is missing and every key that is unexpected.
/// let mut other = ParamStore::new();
/// let other_net = Sequential::new()
///     .push(Relu)
///     .push(Linear::new(&mut other, 2, 3, &mut Rng::seed(0)));
/// let err = load(&path, &other_net, &mut other).unwrap_err();
/// assert!(err.to_string().contains("missing keys (2): 1.weight, 1.bias"));
/// assert!(err.to_string().contains("unexpected keys (2): 0.bias, 0.weight"));
/// std::fs::remove_file(&path).ok();
/// # Ok::<(), Error>(())
/// ```
pub fn load(
    path: impl AsRef<Path>,
    model: &Sequential<f32>,
    params: &mut ParamStore<f32>,
) -> Result<(), Error> {
    load_with(path, model, params, LoadOptions::default()).map(|_| ())
}

/// Like [`load`], but keys in the file that the model does not have are ignored, and
/// returned: the result is those keys, sorted, so the caller can print or check what was
/// dropped (for example the slopes of a `Prelu` loaded into a network that uses `Relu`).
/// Every key the model needs must still be present with its exact shape and the `F32` type,
/// and nothing is written unless all of them check out.
///
/// # Errors
///
/// The errors of [`load`], except that unexpected keys are not an error.
///
/// # Panics
///
/// Panics if `model` was built from a different [`ParamStore`] than `params`.
///
/// ```
/// use cavitation_core::{ParamStore, Rng};
/// use cavitation_nn::{Linear, Prelu, Relu, Sequential, load_lenient, save};
///
/// // A file from a network with PReLU slopes at index 1.
/// let mut file_params = ParamStore::new();
/// let mut rng = Rng::seed(0);
/// let with_prelu = Sequential::new()
///     .push(Linear::new(&mut file_params, 2, 3, &mut rng))
///     .push(Prelu::new(&mut file_params, 1))
///     .push(Linear::new(&mut file_params, 3, 1, &mut rng));
/// let path = std::env::temp_dir()
///     .join(format!("cavitation-doc-lenient-{}.safetensors", std::process::id()));
/// save(&path, &with_prelu, &file_params)?;
///
/// // The same network with ReLU: it has no slopes, so the file's `1.weight` is ignored.
/// let mut params = ParamStore::new();
/// let mut rng = Rng::seed(5);
/// let net = Sequential::new()
///     .push(Linear::new(&mut params, 2, 3, &mut rng))
///     .push(Relu)
///     .push(Linear::new(&mut params, 3, 1, &mut rng));
/// let ignored = load_lenient(&path, &net, &mut params)?;
/// assert_eq!(ignored, ["1.weight"]);
///
/// // The Linear layers hold the file's values.
/// let loaded = params.value(net.parameter_names()[0].1).data().to_vec();
/// let saved = file_params.value(with_prelu.parameter_names()[0].1).data().to_vec();
/// assert_eq!(loaded, saved);
/// std::fs::remove_file(&path).ok();
/// # Ok::<(), cavitation_core::Error>(())
/// ```
pub fn load_lenient(
    path: impl AsRef<Path>,
    model: &Sequential<f32>,
    params: &mut ParamStore<f32>,
) -> Result<Vec<String>, Error> {
    let options = LoadOptions {
        strict: false,
        ..LoadOptions::default()
    };
    load_with(path, model, params, options)
}

/// How [`load_with`] matches the keys of a file to the keys of the model.
///
/// `rename` turns each key of the file into the key the model uses, for files whose keys
/// differ from the model's by a wrapper prefix or by shifted indices. `strict` makes a key
/// that the model lacks, after renaming, an error. The default is no rename and strict,
/// which is what [`load`] does.
///
/// Two ready-made cases: [`LoadOptions::strip_prefix`] for a file saved from a PyTorch
/// `DataParallel` or `DistributedDataParallel` model (every key starts with `module.`), and a
/// closure for a model whose layers sit at different indices in PyTorch, for example because
/// it has `Dropout` layers that this model leaves out.
///
/// ```
/// use cavitation_nn::LoadOptions;
///
/// let default = LoadOptions::default();
/// assert!(default.rename.is_none() && default.strict);
///
/// // Every key of a `DataParallel` file starts with `module.`; this removes it.
/// let options = LoadOptions::strip_prefix("module.");
/// assert!(options.rename.is_some());
/// ```
pub struct LoadOptions<'a> {
    /// Maps each key in the file to the key the model uses; `None` keeps the keys as they are.
    #[allow(clippy::type_complexity)]
    pub rename: Option<Box<dyn Fn(&str) -> String + 'a>>,
    /// Whether a key that the model lacks (after renaming) is an error, listed in
    /// [`KeyErrors::unexpected`] (`true`), or ignored (`false`).
    pub strict: bool,
}

impl fmt::Debug for LoadOptions<'_> {
    /// Shows `strict` and whether `rename` is set (a closure has no text to show).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoadOptions")
            .field("rename", &self.rename.is_some())
            .field("strict", &self.strict)
            .finish()
    }
}

impl Default for LoadOptions<'_> {
    /// No rename, strict: what [`load`] does.
    fn default() -> Self {
        LoadOptions {
            rename: None,
            strict: true,
        }
    }
}

impl<'a> LoadOptions<'a> {
    /// Options that remove `prefix` from the start of each key in the file; a key without
    /// the prefix is kept as it is. Strict.
    ///
    /// ```
    /// use cavitation_nn::LoadOptions;
    ///
    /// let options = LoadOptions::strip_prefix("module.");
    /// let rename = options.rename.unwrap();
    /// assert_eq!(rename("module.0.weight"), "0.weight");
    /// assert_eq!(rename("0.weight"), "0.weight"); // no prefix: kept as it is
    /// ```
    pub fn strip_prefix(prefix: &'a str) -> LoadOptions<'a> {
        LoadOptions {
            rename: Some(Box::new(move |key| {
                key.strip_prefix(prefix).unwrap_or(key).to_string()
            })),
            strict: true,
        }
    }
}

/// Like [`load`], with the key matching set by `options`: each key of the file is first
/// renamed by `options.rename`, then the model's keys are looked up among the renamed ones,
/// and with `options.strict` a renamed key the model lacks is an error. Nothing is written
/// into `params` unless every check passes, a collision between renamed keys included.
///
/// Returns the keys of the file that the model does not have, renamed and sorted, when
/// `options.strict` is `false`; it is empty when `strict` is `true`, since such a key is an
/// error then.
///
/// # Errors
///
/// The errors of [`load`] (the keys in [`Error::Keys`] are the renamed ones), and [`Error::Format`]
/// when two keys of the file rename to the same key; the message names both. For two keys that
/// collide, the first in alphabetical order is named first.
///
/// # Panics
///
/// Panics if `model` was built from a different [`ParamStore`] than `params`.
///
/// ```
/// use cavitation_core::{ParamStore, Rng};
/// use cavitation_nn::{Linear, LoadOptions, Relu, Sequential, load, load_with, save};
///
/// // A file from a PyTorch model that has a `Dropout` after each ReLU. Relu layers stand in
/// // for the Dropouts here, since they have no parameters either: the Linear layers sit at
/// // indices 0, 3 and 6, so the file's keys are `0.*`, `3.*` and `6.*`.
/// let mut file_params = ParamStore::new();
/// let mut rng = Rng::seed(0);
/// let in_file = Sequential::new()
///     .push(Linear::new(&mut file_params, 2, 3, &mut rng))
///     .push(Relu)
///     .push(Relu)
///     .push(Linear::new(&mut file_params, 3, 3, &mut rng))
///     .push(Relu)
///     .push(Relu)
///     .push(Linear::new(&mut file_params, 3, 1, &mut rng));
/// let path = std::env::temp_dir()
///     .join(format!("cavitation-doc-load-with-{}.safetensors", std::process::id()));
/// save(&path, &in_file, &file_params)?;
///
/// // This model leaves the Dropouts out, so its Linear layers are at 0, 2 and 4.
/// let mut params = ParamStore::new();
/// let mut rng = Rng::seed(1);
/// let net = Sequential::new()
///     .push(Linear::new(&mut params, 2, 3, &mut rng))
///     .push(Relu)
///     .push(Linear::new(&mut params, 3, 3, &mut rng))
///     .push(Relu)
///     .push(Linear::new(&mut params, 3, 1, &mut rng));
/// assert!(load(&path, &net, &mut params).is_err()); // the keys `3.*` and `6.*` do not match
///
/// // A closure renumbers: 3 becomes 2 and 6 becomes 4.
/// let renumber = |key: &str| {
///     if let Some(rest) = key.strip_prefix("3.") {
///         format!("2.{rest}")
///     } else if let Some(rest) = key.strip_prefix("6.") {
///         format!("4.{rest}")
///     } else {
///         key.to_string()
///     }
/// };
/// let options = LoadOptions { rename: Some(Box::new(renumber)), ..LoadOptions::default() };
/// load_with(&path, &net, &mut params, options)?;
///
/// // The last Linear of the model now holds the values of the file's `6.weight`.
/// let loaded = params.value(net.parameter_names()[4].1).data().to_vec();
/// let saved = file_params.value(in_file.parameter_names()[4].1).data().to_vec();
/// assert_eq!(loaded, saved);
/// std::fs::remove_file(&path).ok();
/// # Ok::<(), cavitation_core::Error>(())
/// ```
pub fn load_with(
    path: impl AsRef<Path>,
    model: &Sequential<f32>,
    params: &mut ParamStore<f32>,
    options: LoadOptions<'_>,
) -> Result<Vec<String>, Error> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    check_not_torch_save(path, &bytes)?;
    let file = SafeTensors::deserialize(&bytes).map_err(|e| convert(path, e))?;
    let names = model.parameter_names();

    // Renamed key -> key in the file, over every key of the file, so a collision is found
    // before anything is read from a tensor.
    let mut file_keys: Vec<&str> = file.names().into_iter().collect();
    file_keys.sort_unstable();
    let mut by_name: HashMap<String, &str> = HashMap::with_capacity(file_keys.len());
    for key in file_keys {
        let renamed = match &options.rename {
            Some(rename) => rename(key),
            None => key.to_string(),
        };
        if let Some(first) = by_name.get(&renamed) {
            return Err(Error::Format(format!(
                "{}: the keys {first} and {key} both map to {renamed}",
                path.display()
            )));
        }
        by_name.insert(renamed, key);
    }

    // Every key is checked and every problem collected, so one error shows all of them.
    let mut problems = KeyErrors::default();
    let mut staged: Vec<(ParamId, Vec<f32>)> = Vec::with_capacity(names.len());
    for (name, id) in &names {
        let Some(tensor) = by_name.get(name).and_then(|key| file.tensor(key).ok()) else {
            problems.missing.push(name.clone());
            continue;
        };
        if tensor.dtype() != Dtype::F32 {
            problems
                .wrong_dtype
                .push((name.clone(), format!("{:?}", tensor.dtype())));
            continue;
        }
        let expected = params.value(*id).shape().dims();
        if tensor.shape() != expected {
            problems
                .wrong_shape
                .push((name.clone(), expected.to_vec(), tensor.shape().to_vec()));
            continue;
        }
        let values = tensor
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        staged.push((*id, values));
    }
    let mut extra: Vec<String> = by_name
        .keys()
        .filter(|renamed| !names.iter().any(|(name, _)| name == *renamed))
        .cloned()
        .collect();
    extra.sort_unstable();
    if options.strict {
        problems.unexpected = extra.clone();
    }
    if !(problems.missing.is_empty()
        && problems.unexpected.is_empty()
        && problems.wrong_shape.is_empty()
        && problems.wrong_dtype.is_empty())
    {
        // When no key of the model was found at all, show what the file holds instead.
        if problems.missing.len() == names.len() {
            let mut held: Vec<&String> = by_name.keys().collect();
            held.sort_unstable();
            problems.file_keys = held.into_iter().take(5).cloned().collect();
        }
        return Err(Error::Keys(Box::new(problems)));
    }

    for (id, values) in staged {
        params.value_mut(id).data_mut().copy_from_slice(&values);
    }
    Ok(if options.strict { Vec::new() } else { extra })
}

#[cfg(test)]
mod tests {
    //! Tests of `save`, `load`, `load_lenient` and `load_with`.

    use std::path::{Path, PathBuf};

    use cavitation_core::{Error, Graph, KeyErrors, ParamStore, Rng, Tensor};
    use safetensors::tensor::{Dtype, TensorView, serialize_to_file};
    use serde_json::Value;

    use super::{LoadOptions, load, load_lenient, load_with, save};
    use crate::{Linear, Prelu, Relu, Sequential};

    /// A path in the temporary folder that no other test or process uses.
    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "cavitation-io-{}-{name}.safetensors",
            std::process::id()
        ))
    }

    /// `Linear(4, 8), ReLU, Linear(8, 3)` with random weights, the shape of the fixture.
    fn mlp(seed: u64) -> (Sequential<f32>, ParamStore<f32>) {
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(seed);
        let net = Sequential::new()
            .push(Linear::new(&mut params, 4, 8, &mut rng))
            .push(Relu)
            .push(Linear::new(&mut params, 8, 3, &mut rng));
        (net, params)
    }

    /// Every parameter value as raw bits, in the model's parameter order.
    fn bits(net: &Sequential<f32>, params: &ParamStore<f32>) -> Vec<Vec<u32>> {
        net.parameter_names()
            .iter()
            .map(|(_, id)| {
                params
                    .value(*id)
                    .data()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect()
            })
            .collect()
    }

    /// One tensor to write into a test file.
    struct Entry {
        /// The key.
        name: &'static str,
        /// The element type.
        dtype: Dtype,
        /// The dimensions.
        shape: Vec<usize>,
        /// The raw little-endian bytes.
        bytes: Vec<u8>,
    }

    /// An f32 entry of zeros.
    fn zeros(name: &'static str, shape: &[usize]) -> Entry {
        let n: usize = shape.iter().product();
        Entry {
            name,
            dtype: Dtype::F32,
            shape: shape.to_vec(),
            bytes: vec![0; 4 * n],
        }
    }

    /// Writes the entries as a safetensors file.
    fn write_entries(path: &Path, entries: &[Entry]) {
        let views: Vec<(&str, TensorView<'_>)> = entries
            .iter()
            .map(|e| {
                (
                    e.name,
                    TensorView::new(e.dtype, e.shape.clone(), &e.bytes).unwrap(),
                )
            })
            .collect();
        serialize_to_file(views, None, path).unwrap();
    }

    /// The four tensors of the MLP, all zeros, as a base for files with one thing changed.
    fn good_entries() -> Vec<Entry> {
        vec![
            zeros("0.weight", &[8, 4]),
            zeros("0.bias", &[8]),
            zeros("2.weight", &[3, 8]),
            zeros("2.bias", &[3]),
        ]
    }

    /// PyTorch's saved MLP loads by name and gives PyTorch's outputs on the fixture inputs,
    /// within the layer-fixture bound `2 * (k + 2) * 1.19e-7 * sum|terms|`, where a layer's
    /// sum has `k = inputs + 1` terms (the bias is one), so `2 * (inputs + 3) * 1.19e-7 *
    /// sum|terms|`: each layer's bound, plus the first layer's bound carried through the
    /// second layer's weights.
    #[test]
    fn pytorch_state_dict_loads_and_predicts() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/state-dict");
        let (net, mut params) = mlp(7);
        load(dir.join("mlp.safetensors"), &net, &mut params).expect("load");

        let text = std::fs::read_to_string(dir.join("mlp-io.json")).unwrap();
        let v: Value = serde_json::from_str(&text).unwrap();
        let floats = |t: &Value| -> Vec<f64> {
            t["data"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        };
        let x32: Vec<f32> = floats(&v["input"]).iter().map(|&x| x as f32).collect();
        let x64: Vec<f64> = x32.iter().map(|&x| f64::from(x)).collect();
        let torch64 = floats(&v["output_f64"]);
        let (rows, n_in, hidden, n_out) = (5, 4, 8, 3);

        let mut graph = Graph::new();
        let ours: Vec<f32> = {
            let mut tape = graph.tape(&params.values);
            let x = tape.input(Tensor::from_vec(x32, &[rows, n_in]).unwrap().view());
            let y = net.forward(&mut tape, x);
            tape.value(y).data().to_vec()
        };

        let names = net.parameter_names();
        let get = |i: usize| -> Vec<f64> {
            params
                .value(names[i].1)
                .data()
                .iter()
                .map(|&w| f64::from(w))
                .collect()
        };
        let (w1, b1, w2, b2) = (get(0), get(1), get(2), get(3));
        let eps = 1.19e-7;
        for r in 0..rows {
            let mut hidden_err = vec![0.0; hidden];
            for j in 0..hidden {
                let mut sum = b1[j].abs();
                for p in 0..n_in {
                    sum += (x64[r * n_in + p] * w1[j * n_in + p]).abs();
                }
                hidden_err[j] = 2.0 * (n_in as f64 + 3.0) * eps * sum;
            }
            for j in 0..n_out {
                let mut limit = 1e-30;
                let mut sum = b2[j].abs();
                for p in 0..hidden {
                    let pre: f64 = b1[p]
                        + (0..n_in)
                            .map(|q| x64[r * n_in + q] * w1[p * n_in + q])
                            .sum::<f64>();
                    let h = pre.max(0.0);
                    sum += (h * w2[j * hidden + p]).abs();
                    limit += w2[j * hidden + p].abs() * hidden_err[p];
                }
                limit += 2.0 * (hidden as f64 + 3.0) * eps * sum;
                let diff = (f64::from(ours[r * n_out + j]) - torch64[r * n_out + j]).abs();
                assert!(
                    diff <= limit,
                    "row {r} output {j}: diff {diff:e} > {limit:e}"
                );
            }
        }
    }

    /// Saving then loading into a differently initialised model restores every bit.
    #[test]
    fn save_load_roundtrip_bit_identical() {
        let (net, params) = mlp(1);
        let path = temp_path("roundtrip");
        save(&path, &net, &params).unwrap();
        let (net2, mut params2) = mlp(2);
        assert_ne!(bits(&net, &params), bits(&net2, &params2));
        load(&path, &net2, &mut params2).unwrap();
        assert_eq!(bits(&net, &params), bits(&net2, &params2));
        std::fs::remove_file(&path).unwrap();
    }

    /// The key problems of a load error, failing the test for any other error.
    fn keys_of(err: &Error) -> &KeyErrors {
        match err {
            Error::Keys(keys) => keys,
            other => panic!("expected Error::Keys, got {other:?}"),
        }
    }

    /// A file without `2.bias` lists exactly that key as missing.
    #[test]
    fn missing_key_error() {
        let path = temp_path("missing");
        let mut entries = good_entries();
        entries.pop();
        write_entries(&path, &entries);
        let (net, mut params) = mlp(1);
        let err = load(&path, &net, &mut params).unwrap_err();
        let keys = keys_of(&err);
        assert_eq!(keys.missing, ["2.bias"]);
        assert!(keys.unexpected.is_empty() && keys.wrong_shape.is_empty());
        assert!(err.to_string().contains("2.bias"), "{err}");
        std::fs::remove_file(&path).unwrap();
    }

    /// A key the model does not have is an error, except under `load_lenient`.
    #[test]
    fn extra_key_error_unless_lenient() {
        let path = temp_path("extra");
        let mut entries = good_entries();
        entries.push(zeros("3.weight", &[2, 2]));
        write_entries(&path, &entries);
        let (net, mut params) = mlp(1);
        let err = load(&path, &net, &mut params).unwrap_err();
        assert_eq!(keys_of(&err).unexpected, ["3.weight"]);
        let ignored = load_lenient(&path, &net, &mut params).unwrap();
        assert_eq!(ignored, ["3.weight"]);
        assert!(
            params
                .value(net.parameter_names()[0].1)
                .data()
                .iter()
                .all(|&w| w == 0.0)
        );
        std::fs::remove_file(&path).unwrap();
    }

    /// A transposed weight is an error whose message names the key and both shapes.
    #[test]
    fn wrong_shape_error_names_key_and_shapes() {
        let path = temp_path("shape");
        let mut entries = good_entries();
        entries[0] = zeros("0.weight", &[4, 8]);
        write_entries(&path, &entries);
        let (net, mut params) = mlp(1);
        let err = load(&path, &net, &mut params).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("0.weight"), "{text}");
        assert!(text.contains("[8, 4]") && text.contains("[4, 8]"), "{text}");
        std::fs::remove_file(&path).unwrap();
    }

    /// An f16 tensor is a dtype error naming the key.
    #[test]
    fn f16_tensor_is_dtype_error() {
        let path = temp_path("f16");
        let mut entries = good_entries();
        entries[1] = Entry {
            name: "0.bias",
            dtype: Dtype::F16,
            shape: vec![8],
            bytes: vec![0; 16],
        };
        write_entries(&path, &entries);
        let (net, mut params) = mlp(1);
        let err = load(&path, &net, &mut params).unwrap_err();
        assert_eq!(
            keys_of(&err).wrong_dtype,
            [("0.bias".to_string(), "F16".to_string())]
        );
        assert!(err.to_string().contains("F32"), "{err}");
        std::fs::remove_file(&path).unwrap();
    }

    /// A file cut short is a format error.
    #[test]
    fn truncated_file_is_format_error() {
        let (net, mut params) = mlp(1);
        let path = temp_path("truncated");
        save(&path, &net, &params).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 10]).unwrap();
        let err = load(&path, &net, &mut params).unwrap_err();
        assert!(matches!(err, Error::Format(_)), "{err:?}");
        std::fs::remove_file(&path).unwrap();
    }

    /// An error found at the last key leaves every parameter untouched, so the keys before
    /// it are not written as they are read.
    #[test]
    fn failed_load_changes_nothing() {
        let path = temp_path("atomic");
        let mut entries = good_entries();
        for e in entries.iter_mut() {
            e.bytes = 1.5f32.to_le_bytes().repeat(e.bytes.len() / 4);
        }
        entries[3] = zeros("2.bias", &[4]);
        write_entries(&path, &entries);
        let (net, mut params) = mlp(1);
        let before = bits(&net, &params);
        assert!(load(&path, &net, &mut params).is_err());
        assert_eq!(bits(&net, &params), before);
        std::fs::remove_file(&path).unwrap();
    }

    /// The 64 -> 256 -> ReLU -> 128 -> ReLU -> 10 network of the known-result test.
    fn digits_net() -> (Sequential<f32>, ParamStore<f32>) {
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(1);
        let net = Sequential::new()
            .push(Linear::new(&mut params, 64, 256, &mut rng))
            .push(Relu)
            .push(Linear::new(&mut params, 256, 128, &mut rng))
            .push(Relu)
            .push(Linear::new(&mut params, 128, 10, &mut rng));
        (net, params)
    }

    /// A file named for PyTorch modules `fc1`, `fc2`, `fc3` gives one error that lists every
    /// missing and unexpected key and the file's own first keys.
    #[test]
    fn every_problem_listed() {
        let path = temp_path("fc-names");
        write_entries(
            &path,
            &[
                zeros("fc1.weight", &[256, 64]),
                zeros("fc1.bias", &[256]),
                zeros("fc2.weight", &[128, 256]),
                zeros("fc2.bias", &[128]),
                zeros("fc3.weight", &[10, 128]),
                zeros("fc3.bias", &[10]),
            ],
        );
        let (net, mut params) = digits_net();
        let err = load(&path, &net, &mut params).unwrap_err();
        let keys = keys_of(&err);
        assert_eq!(
            keys.missing,
            [
                "0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias"
            ]
        );
        let mut fc: Vec<String> = ["1", "2", "3"]
            .iter()
            .flat_map(|n| [format!("fc{n}.bias"), format!("fc{n}.weight")])
            .collect();
        fc.sort();
        assert_eq!(keys.unexpected, fc);
        assert_eq!(keys.file_keys[0], "fc1.bias");
        assert_eq!(keys.file_keys.len(), 5);
        let text = err.to_string();
        for key in keys.missing.iter().chain(&keys.unexpected) {
            assert!(text.contains(key.as_str()), "{key} not in {text}");
        }
        assert!(text.contains("fc1.bias"), "{text}");
        std::fs::remove_file(&path).unwrap();
    }

    /// A wrong dtype on one key and a wrong shape on another are both in one error.
    #[test]
    fn wrong_shape_and_dtype_together() {
        let path = temp_path("shape-and-dtype");
        let mut entries = good_entries();
        entries[1] = Entry {
            name: "0.bias",
            dtype: Dtype::F16,
            shape: vec![8],
            bytes: vec![0; 16],
        };
        entries[2] = zeros("2.weight", &[3, 7]);
        write_entries(&path, &entries);
        let (net, mut params) = mlp(1);
        let err = load(&path, &net, &mut params).unwrap_err();
        let keys = keys_of(&err);
        assert_eq!(
            keys.wrong_shape,
            [("2.weight".to_string(), vec![3, 8], vec![3, 7])]
        );
        assert_eq!(
            keys.wrong_dtype,
            [("0.bias".to_string(), "F16".to_string())]
        );
        assert!(keys.missing.is_empty() && keys.unexpected.is_empty());
        let text = err.to_string();
        assert!(
            text.contains("2.weight") && text.contains("[3, 7]"),
            "{text}"
        );
        assert!(text.contains("0.bias") && text.contains("F16"), "{text}");
        assert!(!text.contains("transposed"), "{text}");
        std::fs::remove_file(&path).unwrap();
    }

    /// A weight stored as `[inputs, outputs]` gets the transposed hint.
    #[test]
    fn transposed_weight_hint() {
        let path = temp_path("transposed");
        write_entries(
            &path,
            &[
                zeros("0.weight", &[64, 256]),
                zeros("0.bias", &[256]),
                zeros("2.weight", &[128, 256]),
                zeros("2.bias", &[128]),
                zeros("4.weight", &[10, 128]),
                zeros("4.bias", &[10]),
            ],
        );
        let (net, mut params) = digits_net();
        let err = load(&path, &net, &mut params).unwrap_err();
        let text = err.to_string();
        assert!(
            text.contains(
                "this looks transposed: PyTorch and Cavitation store Linear weights as [outputs, inputs]"
            ),
            "{text}"
        );
        assert!(
            text.contains("[64, 256]") && text.contains("[256, 64]"),
            "{text}"
        );
        std::fs::remove_file(&path).unwrap();
    }

    /// With eleven missing keys, ten are listed and the message ends with "and 1 more".
    #[test]
    fn eleven_missing_keys_capped_at_ten() {
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(1);
        let mut net = Sequential::new();
        for _ in 0..5 {
            net = net.push(Linear::new(&mut params, 2, 2, &mut rng));
        }
        net = net.push(Prelu::new(&mut params, 1));
        let path = temp_path("eleven");
        write_entries(&path, &[zeros("other", &[1])]);
        let err = load(&path, &net, &mut params).unwrap_err();
        let keys = keys_of(&err);
        assert_eq!(keys.missing.len(), 11);
        let text = err.to_string();
        assert!(text.contains("and 1 more"), "{text}");
        assert!(text.contains(keys.missing[9].as_str()), "{text}");
        assert!(!text.contains(keys.missing[10].as_str()), "{text}");
        std::fs::remove_file(&path).unwrap();
    }

    /// A PReLU network's file loaded into the ReLU network reports the slope key it dropped.
    #[test]
    fn lenient_returns_ignored_keys() {
        let mut file_params = ParamStore::new();
        let mut rng = Rng::seed(0);
        let with_prelu = Sequential::new()
            .push(Linear::new(&mut file_params, 4, 8, &mut rng))
            .push(Prelu::new(&mut file_params, 1))
            .push(Linear::new(&mut file_params, 8, 3, &mut rng));
        let path = temp_path("prelu");
        save(&path, &with_prelu, &file_params).unwrap();
        let (net, mut params) = mlp(1);
        assert!(load(&path, &net, &mut params).is_err());
        let ignored = load_lenient(&path, &net, &mut params).unwrap();
        assert_eq!(ignored, ["1.weight"]);
        std::fs::remove_file(&path).unwrap();
    }

    /// Writes `bytes` as a file and loads it into the MLP, failing the test if `load` panics
    /// or if it returns an error after changing a parameter.
    fn load_bytes(name: &str, bytes: &[u8]) -> Result<(), Error> {
        let path = temp_path(name);
        std::fs::write(&path, bytes).unwrap();
        let (net, mut params) = mlp(1);
        let before = bits(&net, &params);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            load(&path, &net, &mut params)
        }));
        std::fs::remove_file(&path).unwrap();
        let result = result.unwrap_or_else(|_| panic!("{name}: load panicked"));
        if result.is_err() {
            assert_eq!(bits(&net, &params), before, "{name}: a failed load wrote");
        }
        result
    }

    /// A file made of an 8-byte little-endian header length, the header and `data`.
    fn with_header(header: &str, data: &[u8]) -> Vec<u8> {
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(data);
        bytes
    }

    /// The header of `good_entries` with `first` as the entry for `0.weight`.
    fn header_with(first: &str) -> String {
        format!(
            r#"{{"0.weight":{first},"0.bias":{{"dtype":"F32","shape":[8],"data_offsets":[128,160]}},"2.weight":{{"dtype":"F32","shape":[3,8],"data_offsets":[160,256]}},"2.bias":{{"dtype":"F32","shape":[3],"data_offsets":[256,268]}}}}"#
        )
    }

    /// Malformed files are format errors, never panics, and change nothing: a length prefix
    /// cut short, a header length past the end of the file (or near `u64::MAX`), a header
    /// that is not JSON, an element type that does not exist, and data offsets out of range
    /// or not matching the shape. The well-formed header these are made from loads.
    #[test]
    fn malformed_headers_are_format_errors() {
        let good = r#"{"dtype":"F32","shape":[8,4],"data_offsets":[0,128]}"#;
        let data = vec![0u8; 268];
        load_bytes("good-header", &with_header(&header_with(good), &data)).unwrap();

        let mut past_end = with_header(&header_with(good), &data);
        past_end[..8].copy_from_slice(&10_000u64.to_le_bytes());
        let mut huge = past_end.clone();
        huge[..8].copy_from_slice(&(u64::MAX - 3).to_le_bytes());
        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("empty", Vec::new()),
            ("short-length", vec![12, 0, 0]),
            ("past-end", past_end),
            ("huge-length", huge),
            ("not-json", with_header("{not json", &data)),
            (
                "unknown-dtype",
                with_header(
                    &header_with(r#"{"dtype":"Q7","shape":[8,4],"data_offsets":[0,128]}"#),
                    &data,
                ),
            ),
            (
                "offsets-past-data",
                with_header(
                    &header_with(r#"{"dtype":"F32","shape":[8,4],"data_offsets":[0,99999]}"#),
                    &data,
                ),
            ),
            (
                "offsets-reversed",
                with_header(
                    &header_with(r#"{"dtype":"F32","shape":[8,4],"data_offsets":[128,0]}"#),
                    &data,
                ),
            ),
            (
                "offsets-not-the-shape",
                with_header(
                    &header_with(r#"{"dtype":"F32","shape":[8,4],"data_offsets":[0,4]}"#),
                    &data,
                ),
            ),
        ];
        for (name, bytes) in cases {
            let err = load_bytes(name, &bytes).unwrap_err();
            assert!(matches!(err, Error::Format(_)), "{name}: {err:?}");
        }
    }

    /// Writes `bytes` to a fresh path, loads it into the MLP and returns the error.
    fn load_error_of(name: &str, bytes: &[u8]) -> Error {
        load_bytes(name, bytes).unwrap_err()
    }

    /// A zip archive, which is what `torch.save` writes by default, is named as such.
    #[test]
    fn zip_file_is_recognised() {
        let mut bytes = b"PK\x03\x04".to_vec();
        bytes.extend_from_slice(&[0; 60]);
        let err = load_error_of("zip", &bytes);
        assert!(
            matches!(&err, Error::Format(m) if m.contains("torch.save") && m.contains("zip")),
            "{err:?}"
        );
    }

    /// A pickle stream, the legacy `torch.save` format, is named as such.
    #[test]
    fn pickle_file_is_recognised() {
        let mut bytes = vec![0x80, 0x02];
        bytes.extend_from_slice(&[0; 60]);
        let err = load_error_of("pickle", &bytes);
        assert!(
            matches!(&err, Error::Format(m) if m.contains("torch.save")),
            "{err:?}"
        );
    }

    /// A safetensors file whose header is exactly 128 bytes starts with the byte 0x80 and is
    /// not mistaken for a pickle.
    #[test]
    fn header_of_128_bytes_is_not_a_pickle() {
        let header = format!("{{}}{}", " ".repeat(126));
        let path = temp_path("header-128");
        std::fs::write(&path, with_header(&header, &[])).unwrap();
        let net = Sequential::<f32>::new();
        let mut params = ParamStore::new();
        load(&path, &net, &mut params).unwrap();
        std::fs::remove_file(&path).unwrap();
    }

    /// Empty, 1-byte (a lone pickle marker) and 3-byte files are format errors from the
    /// safetensors parser, not panics and not the `torch.save` message.
    #[test]
    fn tiny_files_are_format_errors() {
        for (name, bytes) in [
            ("tiny-0", vec![]),
            ("tiny-1", vec![0x80]),
            ("tiny-3", vec![b'P', b'K', 3]),
        ] {
            let err = load_error_of(name, &bytes);
            assert!(
                matches!(&err, Error::Format(m) if !m.contains("torch.save")),
                "{name}: {err:?}"
            );
        }
    }

    /// The MLP of seed 3 saved with `prefix` before each key, returned with its parameters.
    fn save_prefixed(path: &Path, prefix: &str) -> (Sequential<f32>, ParamStore<f32>) {
        let (net, params) = mlp(3);
        let names = net.parameter_names();
        let keys: Vec<String> = names.iter().map(|(n, _)| format!("{prefix}{n}")).collect();
        let buffers: Vec<Vec<u8>> = names
            .iter()
            .map(|(_, id)| {
                params
                    .value(*id)
                    .data()
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect()
            })
            .collect();
        let views: Vec<(&str, TensorView<'_>)> = names
            .iter()
            .zip(&keys)
            .zip(&buffers)
            .map(|(((_, id), key), bytes)| {
                let shape = params.value(*id).shape().dims().to_vec();
                (
                    key.as_str(),
                    TensorView::new(Dtype::F32, shape, bytes).unwrap(),
                )
            })
            .collect();
        serialize_to_file(views, None, path).unwrap();
        (net, params)
    }

    /// A `DataParallel` file, every key starting with `module.`, loads through
    /// `strip_prefix` and gives the saved values; plain `load` rejects it.
    #[test]
    fn module_prefix_loads() {
        let path = temp_path("module-prefix");
        let (net, saved) = save_prefixed(&path, "module.");
        let (net2, mut params) = mlp(4);
        assert!(load(&path, &net2, &mut params).is_err());
        load_with(
            &path,
            &net2,
            &mut params,
            LoadOptions::strip_prefix("module."),
        )
        .unwrap();
        assert_eq!(bits(&net, &saved), bits(&net2, &params));
        std::fs::remove_file(&path).unwrap();
    }

    /// `strip_prefix` keeps keys without the prefix as they are.
    #[test]
    fn strip_prefix_keeps_other_keys() {
        let path = temp_path("module-mixed");
        let mut entries = good_entries();
        entries[0].name = "module.0.weight";
        write_entries(&path, &entries);
        let (net, mut params) = mlp(1);
        load_with(
            &path,
            &net,
            &mut params,
            LoadOptions::strip_prefix("module."),
        )
        .unwrap();
        std::fs::remove_file(&path).unwrap();
    }

    /// A file saved from a network with a Dropout at index 2 and 5 (keys `0.*`, `3.*`,
    /// `6.*`) loads into the network without them with a closure that renumbers the keys.
    #[test]
    fn dropout_shifted_indices_load() {
        let path = temp_path("dropout-shift");
        write_entries(
            &path,
            &[
                Entry {
                    bytes: 1.0f32.to_le_bytes().repeat(32),
                    ..zeros("0.weight", &[8, 4])
                },
                zeros("0.bias", &[8]),
                zeros("3.weight", &[8, 8]),
                zeros("3.bias", &[8]),
                zeros("6.weight", &[3, 8]),
                zeros("6.bias", &[3]),
            ],
        );
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(5);
        let net = Sequential::new()
            .push(Linear::new(&mut params, 4, 8, &mut rng))
            .push(Relu)
            .push(Linear::new(&mut params, 8, 8, &mut rng))
            .push(Relu)
            .push(Linear::new(&mut params, 8, 3, &mut rng));
        let rename = |key: &str| {
            if let Some(rest) = key.strip_prefix("3.") {
                format!("2.{rest}")
            } else if let Some(rest) = key.strip_prefix("6.") {
                format!("4.{rest}")
            } else {
                key.to_string()
            }
        };
        let options = LoadOptions {
            rename: Some(Box::new(rename)),
            ..LoadOptions::default()
        };
        load_with(&path, &net, &mut params, options).unwrap();
        let names = net.parameter_names();
        let first = params.value(names[0].1).data();
        assert_eq!(first.len(), 32);
        assert!(first.iter().all(|&w| w == 1.0));
        assert!(params.value(names[2].1).data().iter().all(|&w| w == 0.0));
        std::fs::remove_file(&path).unwrap();
    }

    /// Two file keys renamed to one key are an error naming both and the key, and nothing
    /// is written.
    #[test]
    fn rename_collision_is_an_error() {
        let path = temp_path("collision");
        write_entries(&path, &good_entries());
        let (net, mut params) = mlp(1);
        let before = bits(&net, &params);
        let options = LoadOptions {
            rename: Some(Box::new(|_| "0.weight".to_string())),
            ..LoadOptions::default()
        };
        let err = load_with(&path, &net, &mut params, options).unwrap_err();
        let want = format!(
            "{}: the keys 0.bias and 0.weight both map to 0.weight",
            path.display()
        );
        assert!(matches!(&err, Error::Format(m) if *m == want), "{err:?}");
        assert_eq!(bits(&net, &params), before);
        std::fs::remove_file(&path).unwrap();
    }

    /// With a rename, `strict` checks the renamed keys: a key mapped to a name the model
    /// lacks is listed in `KeyErrors::unexpected` under the renamed key, and ignored (and
    /// returned) when not strict.
    #[test]
    fn strict_checks_renamed_keys() {
        let path = temp_path("renamed-extra");
        let mut entries = good_entries();
        entries.push(zeros("x.weight", &[2, 2]));
        write_entries(&path, &entries);
        let (net, mut params) = mlp(1);
        let rename = |key: &str| key.replace("x.", "9.");
        let strict = LoadOptions {
            rename: Some(Box::new(rename)),
            ..LoadOptions::default()
        };
        let err = load_with(&path, &net, &mut params, strict).unwrap_err();
        assert_eq!(keys_of(&err).unexpected, ["9.weight"]);
        let lenient = LoadOptions {
            rename: Some(Box::new(rename)),
            strict: false,
        };
        let ignored = load_with(&path, &net, &mut params, lenient).unwrap();
        assert_eq!(ignored, ["9.weight"]);
        std::fs::remove_file(&path).unwrap();
    }

    /// `Debug` shows whether a rename is set, and `Default` is strict with none.
    #[test]
    fn load_options_debug_and_default() {
        let default = LoadOptions::default();
        assert!(default.rename.is_none() && default.strict);
        assert!(format!("{default:?}").contains("rename: false"));
        assert!(format!("{:?}", LoadOptions::strip_prefix("m.")).contains("rename: true"));
    }

    /// The model behind `from-rust.safetensors`: the fixture's MLP with every parameter set
    /// from its position `k` (counting through the parameters in order) to
    /// `(k * 7919 mod 199) * 0.01 - 1` in f32, so its values come from neither initialiser.
    fn rust_fixture_model() -> (Sequential<f32>, ParamStore<f32>) {
        let (net, mut params) = mlp(0);
        let mut k = 0u32;
        for (_, id) in net.parameter_names() {
            for v in params.value_mut(id).data_mut() {
                *v = (k * 7919 % 199) as f32 * 0.01 - 1.0;
                k += 1;
            }
        }
        (net, params)
    }

    /// A tensor as the fixture JSON writes it: `{"shape": [...], "data": [...]}`.
    fn tensor_json(shape: &[usize], data: &[f32]) -> Value {
        serde_json::json!({ "shape": shape, "data": data })
    }

    /// `save` writes `tests/fixtures/state-dict/from-rust.safetensors` byte for byte, and
    /// `from-rust-io.json` holds the same parameter values, so the PyTorch check of
    /// `scripts/fixtures/make_fixtures.py --check-rust-save` (which loads that file with
    /// `safetensors.torch.load_file`, runs it in PyTorch and compares) covers what `save`
    /// writes today. With `CAVITATION_WRITE_FIXTURES` set, the test writes both files instead
    /// (the JSON with the inputs of `mlp-io.json` and this model's outputs on them); rerun
    /// the PyTorch check after that.
    #[test]
    fn save_writes_the_file_pytorch_was_checked_against() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/state-dict");
        let (net, params) = rust_fixture_model();
        let path = temp_path("from-rust");
        save(&path, &net, &params).unwrap();
        let written = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let names = net.parameter_names();

        if std::env::var_os("CAVITATION_WRITE_FIXTURES").is_some() {
            let io: Value =
                serde_json::from_str(&std::fs::read_to_string(dir.join("mlp-io.json")).unwrap())
                    .unwrap();
            let input: Vec<f32> = io["input"]["data"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect();
            let rows = input.len() / 4;
            let mut graph = Graph::new();
            let output: Vec<f32> = {
                let mut tape = graph.tape(&params.values);
                let x = tape.input(Tensor::from_vec(input.clone(), &[rows, 4]).unwrap().view());
                let y = net.forward(&mut tape, x);
                tape.value(y).data().to_vec()
            };
            let mut parameters = serde_json::Map::new();
            for (name, id) in &names {
                let value = params.value(*id);
                parameters.insert(
                    name.clone(),
                    tensor_json(value.shape().dims(), value.data()),
                );
            }
            let body = serde_json::json!({
                "keys": names.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
                "params": parameters,
                "input": tensor_json(&[rows, 4], &input),
                "output": tensor_json(&[rows, 3], &output),
            });
            std::fs::write(dir.join("from-rust.safetensors"), &written).unwrap();
            std::fs::write(dir.join("from-rust-io.json"), format!("{body}\n")).unwrap();
            return;
        }

        let committed = std::fs::read(dir.join("from-rust.safetensors")).unwrap();
        assert!(
            written == committed,
            "save no longer writes tests/fixtures/state-dict/from-rust.safetensors; if the \
             change is meant, rewrite it with CAVITATION_WRITE_FIXTURES=1 and rerun \
             make_fixtures.py --check-rust-save"
        );
        let io: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("from-rust-io.json")).unwrap())
                .unwrap();
        for (name, id) in &names {
            let stored: Vec<u32> = io["params"][name.as_str()]["data"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| (x.as_f64().unwrap() as f32).to_bits())
                .collect();
            let ours: Vec<u32> = params
                .value(*id)
                .data()
                .iter()
                .map(|v| v.to_bits())
                .collect();
            assert_eq!(stored, ours, "{name}");
        }
    }
}
