use crate::toolbet;
use rand::{Rng, RngExt};

#[derive(Debug)]
pub enum NeuronError {
    InputMismatch,
}

pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl Neuron {
    pub fn new(rng: &mut impl Rng, weights_count: usize) -> Neuron {
        let mut weights = Vec::new();

        for _ in 0..weights_count {
            let weight = rng.random_range(-1.0..1.0);

            weights.push(weight);
        }

        let bias = rng.random_range(-1.0..1.0);

        return Neuron {
            weights: weights,
            bias: bias,
        };
    }

    // This function produces 'z'. It is just the sum of bias and the multiplication of all inputs by weighs
    pub fn calc_z(&self, inputs: &Vec<f64>) -> Result<f64, NeuronError> {
        if inputs.len() != self.weights.len() {
            return Result::Err(NeuronError::InputMismatch);
        }

        let mut result: f64 = 0.0;

        for i in 0..self.weights.len() {
            result += inputs[i] * self.weights[i];
        }

        return Result::Ok(result + self.bias);
    }

    // This function output the a value which result from 'estimulate' the neuron
    pub fn forward(&self, inputs: &Vec<f64>) -> Result<f64, NeuronError> {
        return Result::Ok(toolbet::sigmoid_active(self.calc_z(inputs)?));
    }

    pub fn relu_train(
        &mut self,
        inputs: &Vec<f64>,
        target: f64,
        learning_rate: f64,
    ) -> Result<(), NeuronError> {
        let z = self.calc_z(inputs)?;
        let a = toolbet::relu_activate(z);
        let gradient = (a - target) * toolbet::relu_gradient(z);

        for i in 0..self.weights.len() {
            let weight_gradient = gradient * inputs[i];

            self.weights[i] -= learning_rate * weight_gradient;
        }

        let bias_gradient = gradient;

        self.bias -= learning_rate * bias_gradient;

        return Result::Ok(());
    }

    pub fn sigmoid_train(
        &mut self,
        inputs: &Vec<f64>,
        target: f64,
        learning_rate: f64,
    ) -> Result<(), NeuronError> {
        let z = self.calc_z(inputs)?;
        let a = toolbet::sigmoid_active(z);

        for i in 0..self.weights.len() {
            let weight_gradient = (a - target) * inputs[i];

            self.weights[i] -= learning_rate * weight_gradient;
        }

        let bias_gradient = a - target;

        self.bias -= learning_rate * bias_gradient;

        return Result::Ok(());
    }
}
