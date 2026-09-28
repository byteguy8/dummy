use crate::{
    neuron::{Neuron, NeuronError},
    toolbet,
};
use rand::Rng;

struct NeuronLayer {
    neurons: Vec<Neuron>,
}

impl NeuronLayer {
    pub fn new(rng: &mut impl Rng, weights_count: usize, count: usize) -> NeuronLayer {
        let mut neurons = Vec::new();

        for _ in 0..count {
            let neuron = Neuron::new(rng, weights_count);

            neurons.push(neuron);
        }

        return NeuronLayer { neurons };
    }

    pub fn forward(&self, inputs: &Vec<f64>) -> Result<Vec<f64>, NeuronError> {
        let activations = Vec::new();

        for i in 0..self.neurons.len() {
            let activation = self.neurons[i].forward(inputs)?;
        }

        return Result::Ok(activations);
    }
}
