pub fn squared_error_loss(prediction: f64, target: f64) -> f64 {
    return 0.5 * (prediction - target).powi(2);
}

pub fn binary_cross_entropy(prediction: f64, target: f64) -> f64 {
    -(target * prediction.ln() + (1.0 - target) * (1.0 - prediction).ln())
}

// This function produces 'a'. It implements ReLU (Rectified Linear Unit). ReLU is an activation function. It takes a input
// and decides what comes out of the neuron. ReLU(z) = max(0,z) = { z if z > 0, 0 otherwise }
pub fn relu_activate(z: f64) -> f64 {
    return z.max(0.0);
}

// This function produces 'a'. Sigmoid activation: a = 1 / (1 + e^(-z))
pub fn sigmoid_active(z: f64) -> f64 {
    return 1.0 / (1.0 + (-z).exp());
}

pub fn relu_gradient(z: f64) -> f64 {
    return if z > 0.0 { 1.0 } else { 0.0 };
}

pub fn sigmoid_gradient(a: f64) -> f64 {
    return a * (1.0 - a);
}
