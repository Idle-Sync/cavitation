//! A synthetic steering data set: seeded 1 × 32 × 32 grey images of a road edge, with the
//! edge's angle in degrees as the target, generated in plain Rust.
//!
//! Each image shows a bright road on the right of a straight edge and darker ground on its
//! left. The edge starts at the bottom row, a few pixels either side of the centre, and leans
//! by its angle: 0° runs straight up, a positive angle leans the top of the edge to the right
//! (a road turning right), a negative one to the left. The angle is drawn uniformly from
//! [−30°, 30°]. Each image also gets its own brightness offset and per-pixel noise, so the
//! network has to read the edge rather than a fixed pixel.

use cavitation_core::{Conv2dGeom, ParamStore, Pool2dGeom, Rng, Tensor};
use cavitation_nn::{
    Conv2d, Dataset, FitConfig, Flatten, Health, Linear, Loss, MaxPool2d, OptimizerConfig, Relu,
    Sequential, Targets,
};

/// The height and width of each image.
pub const SIZE: usize = 32;

/// The largest angle, in degrees, either way.
pub const MAX_ANGLE: f64 = 30.0;

/// `n` images from `seed`, as rows of `SIZE · SIZE` pixels in [0, 1] (one channel, row by
/// row), with each image's edge angle in degrees as a `[n, 1]` value target. The same `n`
/// and `seed` always give the same data; the first `m` images of `n` are the images `m`
/// gives.
///
/// ```
/// use cavitation_datasets::steering;
/// use cavitation_nn::Targets;
/// let data = steering::generate(4, 7);
/// assert_eq!(data.x.shape().dims(), [4, 1024]);
/// let Targets::Values(angles) = &data.targets else { unreachable!() };
/// assert!(angles.data().iter().all(|a| a.abs() <= 30.0));
/// assert_eq!(data, steering::generate(4, 7));
/// ```
pub fn generate(n: usize, seed: u64) -> Dataset {
    let mut rng = Rng::seed(seed);
    let mut x = Vec::with_capacity(n * SIZE * SIZE);
    let mut angles = Vec::with_capacity(n);
    for _ in 0..n {
        let angle = rng.range(-MAX_ANGLE, MAX_ANGLE);
        let base = SIZE as f64 / 2.0 + rng.range(-4.0, 4.0);
        let brightness = rng.range(-0.1, 0.1);
        render(angle, base, brightness, 0.05, &mut rng, &mut x);
        angles.push(angle as f32);
    }
    Dataset {
        x: Tensor::from_vec(x, &[n, SIZE * SIZE]).expect("n rows of SIZE² pixels"),
        targets: Targets::Values(Tensor::from_vec(angles, &[n, 1]).expect("n angles")),
    }
}

/// The example's network, with PyTorch's default initialisation from `Rng::seed(seed)`: a
/// 5 × 5 convolution to 8 channels with stride 2 and padding 2 (16 × 16), ReLU, a 3 × 3
/// convolution to 16 channels with stride 2 and padding 1 (8 × 8), ReLU, 2 × 2 max pooling
/// (4 × 4), flatten, `Linear(256 → 32)`, ReLU and `Linear(32 → 1)`, the angle.
///
/// ```
/// use cavitation_core::ParamStore;
/// let mut params = ParamStore::new();
/// let net = cavitation_datasets::steering::network(&mut params, 0);
/// let count: usize = net
///     .parameter_names()
///     .iter()
///     .map(|(_, id)| params.value(*id).data().len())
///     .sum();
/// assert_eq!((net.len(), count), (9, 9_633));
/// ```
pub fn network(params: &mut ParamStore, seed: u64) -> Sequential {
    let mut rng = Rng::seed(seed);
    let g1 = Conv2dGeom::with_options(1, 8, (SIZE, SIZE), (5, 5), (2, 2), (2, 2), (1, 1))
        .expect("a valid convolution");
    let g2 = Conv2dGeom::with_options(8, 16, (16, 16), (3, 3), (2, 2), (1, 1), (1, 1))
        .expect("a valid convolution");
    let pool = Pool2dGeom::new(16, (8, 8), (2, 2)).expect("a valid pooling");
    Sequential::new()
        .push(Conv2d::new(params, g1, &mut rng))
        .push(Relu)
        .push(Conv2d::new(params, g2, &mut rng))
        .push(Relu)
        .push(MaxPool2d::new(pool))
        .push(Flatten)
        .push(Linear::new(params, 256, 32, &mut rng))
        .push(Relu)
        .push(Linear::new(params, 32, 1, &mut rng))
}

/// The example's training recipe for `epochs` epochs (8 in the example): Adam at learning
/// rate 0.002, batch 32, mean squared error on the angle in degrees, shuffle seed 3, and the
/// default health checks without the dead-ReLU report. That report counts each position of
/// a convolution's output as a neuron, and on these images many positions see only flat
/// road or ground, so a ReLU after a convolution outputs 0 there for a whole epoch without
/// being dead.
///
/// ```
/// let config = cavitation_datasets::steering::recipe(8);
/// assert_eq!((config.epochs, config.batch), (8, 32));
/// ```
pub fn recipe(epochs: usize) -> FitConfig<'static> {
    FitConfig {
        epochs,
        batch: 32,
        optimizer: OptimizerConfig::Adam {
            lr: 0.002,
            weight_decay: 0.0,
        },
        loss: Loss::Mse,
        shuffle_seed: Some(3),
        health: Health {
            dead_relu: false,
            ..Health::default()
        },
        ..FitConfig::default()
    }
}

/// Appends one image to `out`: an edge through `(base, SIZE)` on the bottom border leaning
/// `angle` degrees from vertical, ground 0.25 on its left and road 0.75 on its right with a
/// one-pixel ramp between them, shifted by `brightness`, with noise in ±`noise` from `rng`
/// on each pixel, clamped to [0, 1].
fn render(angle: f64, base: f64, brightness: f64, noise: f64, rng: &mut Rng, out: &mut Vec<f32>) {
    let (sin, cos) = angle.to_radians().sin_cos();
    for row in 0..SIZE {
        for col in 0..SIZE {
            let (px, py) = (col as f64 + 0.5, row as f64 + 0.5);
            // Signed distance from the edge, positive on the road side: the edge runs up
            // from the base along (sin, −cos).
            let side = cos * (px - base) + sin * (py - SIZE as f64);
            let road = (0.5 + side).clamp(0.0, 1.0);
            let v = 0.25 + 0.5 * road + brightness + noise * rng.range(-1.0, 1.0);
            out.push(v.clamp(0.0, 1.0) as f32);
        }
    }
}

/// Tests for the steering data: fixed by the seed, a prefix of a longer run, the angle range,
/// and the edge where its angle puts it.
#[cfg(test)]
mod tests {
    use super::*;

    /// The angles of `data`.
    fn angles(data: &Dataset) -> &[f32] {
        let Targets::Values(a) = &data.targets else {
            unreachable!("steering has value targets")
        };
        a.data()
    }

    /// The same seed gives the same data, another seed other data, and a shorter run is a
    /// prefix of a longer one.
    #[test]
    fn seeded_and_prefixed() {
        let a = generate(20, 3);
        assert_eq!(a, generate(20, 3));
        assert_ne!(a.x.data(), generate(20, 4).x.data());
        let short = generate(5, 3);
        assert_eq!(short.x.data(), &a.x.data()[..5 * SIZE * SIZE]);
        assert_eq!(angles(&short), &angles(&a)[..5]);
    }

    /// Every pixel is in [0, 1]; the angles stay within ±30° and spread over the range.
    #[test]
    fn ranges() {
        let data = generate(500, 1);
        assert!(data.x.data().iter().all(|v| (0.0..=1.0).contains(v)));
        let a = angles(&data);
        assert!(a.iter().all(|v| v.abs() <= MAX_ANGLE as f32));
        assert!(a.iter().any(|&v| v < -25.0) && a.iter().any(|&v| v > 25.0));
    }

    /// With no noise or offset, a 0° edge at the centre splits each row into ground on the
    /// left and road on the right; at +20° the top row's edge sits about
    /// `31 · tan 20° ≈ 11.3` pixels right of the bottom row's (their centres are 31 rows
    /// apart), and at −20° as far left.
    #[test]
    fn edge_follows_the_angle() {
        /// Where `row` of the image `img` first reaches the halfway value.
        fn crossing(img: &[f32], row: usize) -> usize {
            (0..SIZE)
                .find(|&c| img[row * SIZE + c] >= 0.5)
                .unwrap_or(SIZE)
        }
        let quiet = |angle| {
            let mut img = Vec::new();
            render(angle, 16.0, 0.0, 0.0, &mut Rng::seed(0), &mut img);
            img
        };
        let straight = quiet(0.0);
        for row in [0, 15, 31] {
            assert_eq!(crossing(&straight, row), 16, "row {row}");
        }
        let want = 31.0 * 20f64.to_radians().tan();
        let right = quiet(20.0);
        let shift = crossing(&right, 0) as f64 - crossing(&right, SIZE - 1) as f64;
        assert!((shift - want).abs() <= 1.0, "{shift}");
        let left = quiet(-20.0);
        let shift = crossing(&left, SIZE - 1) as f64 - crossing(&left, 0) as f64;
        assert!((shift - want).abs() <= 1.0, "{shift}");
    }
}
