mod neuron;
mod neuron_layer;
mod toolbet;

use crate::neuron::Neuron;
use rand::rng;

fn train_neuron(neuron: &mut Neuron, training_data: &[(&Vec<f64>, f64)]) {
    for _ in 0..1000 {
        for &(inputs, target) in training_data {
            neuron.sigmoid_train(inputs, target, 0.01);
        }
    }
}

fn test_neuron(neuron: &Neuron, test_data: &[&Vec<f64>]) {
    for inputs in test_data {
        let output = neuron.forward(*inputs).unwrap();
        let real_output = if output >= 0.5 { 1 } else { 0 };

        println!("{:?} -> {:.4} -> {}", inputs, output, real_output);
    }

    println!("weights: {:?}", neuron.weights);
    println!("bias: {}", neuron.bias);
}

fn main() {
    let mut random = rng();
    let mut or_neuron = Neuron::new(&mut random, 2);
    let mut and_neuron = Neuron::new(&mut random, 2);
    let mut not_neuron = Neuron::new(&mut random, 1);

    let or_training_data = [
        (&vec![0.0, 0.0], 0.0),
        (&vec![0.0, 1.0], 1.0),
        (&vec![1.0, 0.0], 1.0),
        (&vec![1.0, 1.0], 1.0),
    ];
    let and_training_data = [
        (&vec![0.0, 0.0], 0.0),
        (&vec![0.0, 1.0], 0.0),
        (&vec![1.0, 0.0], 0.0),
        (&vec![1.0, 1.0], 1.0),
    ];
    let not_training_data = [(&vec![0.0], 1.0), (&vec![1.0], 0.0)];
    let or_test_data = [
        &vec![0.0, 0.0],
        &vec![0.0, 1.0],
        &vec![1.0, 0.0],
        &vec![1.0, 1.0],
    ];
    let and_test_data = [
        &vec![0.0, 0.0],
        &vec![0.0, 1.0],
        &vec![1.0, 0.0],
        &vec![1.0, 1.0],
    ];
    let not_test_data = [&vec![0.0], &vec![1.0]];

    train_neuron(&mut or_neuron, &or_training_data);
    train_neuron(&mut and_neuron, &and_training_data);
    train_neuron(&mut not_neuron, &not_training_data);
    test_neuron(&or_neuron, &or_test_data);
    test_neuron(&and_neuron, &and_test_data);
    test_neuron(&not_neuron, &not_test_data);
}
