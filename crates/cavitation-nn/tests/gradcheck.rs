//! Gradient checks of the layers, in `f64`.

use cavitation_core::{ParamStore, Rng, Tensor, gradcheck_params};
use cavitation_nn::{Init, Layer, Linear};

/// A `Linear` layer's forward pass has the right gradient for its input, weight and bias.
#[test]
fn gradcheck_linear() {
    let mut params = ParamStore::<f64>::new_generic();
    let mut rng = Rng::seed(11);
    // Non-zero biases, so that the bias gradient is exercised at a generic point.
    let layer = Linear::with_init(
        &mut params,
        3,
        5,
        Init::XavierNormal,
        Init::XavierUniform,
        &mut rng,
    );
    let data = (0..12).map(|i| ((i as f64) * 0.7).sin()).collect();
    let x = Tensor::from_vec(data, &[4, 3]).unwrap();
    let report = gradcheck_params(&mut params, &[x], |tape, v| layer.forward(tape, v[0]));
    report.assert_passed();
}
