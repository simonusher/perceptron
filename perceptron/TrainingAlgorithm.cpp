#include "pch.h"
#include "TrainingAlgorithm.h"
#include <iostream>

typedef std::pair<std::valarray<double>, double> TrainingExample;
typedef std::vector<TrainingExample> TrainingSet;

void TrainingAlgorithm::train(Perceptron & neuron, const TrainingSet& trainingSet, double learningRateAlpha, int iterations) {
	for (int i = 0; i < iterations; i++) {
		std::cout << i << std::endl;
		for (int j = 0; j < trainingSet.size(); j++) {
			trainOneIteration(neuron, trainingSet[j], learningRateAlpha);
		}
	}
}

void TrainingAlgorithm::trainOneIteration(Perceptron& neuron, const TrainingExample& example, double alpha) {
	double output = neuron.calculateOutput(example.first);
	double delta = calculateError(example.second, output);
	std::cout << "\t" << delta << std::endl;
	neuron.learn(alpha * delta * example.first);
}

void TrainingAlgorithm::train(Adaline& neuron, const TrainingSet& trainingSet, double learningRateAlpha, double learningStopErrorThreshold) {
	double meanSquaredError;
	do {
		meanSquaredError = trainOneIterationAdaline(neuron, trainingSet, learningRateAlpha);
	} while (meanSquaredError > learningStopErrorThreshold);
}

double TrainingAlgorithm::trainOneIterationAdaline(Adaline& neuron, const TrainingSet& trainingSet, double alpha) {
	double meanSquaredError = 0;
	for (int i = 0; i < trainingSet.size(); i++) {
		double activation = neuron.forward(trainingSet[i].first);
		double error = trainingSet[i].second - activation;
		meanSquaredError += error * error;
		neuron.learn(2 * error * alpha * trainingSet[i].first);
	}
	meanSquaredError /= trainingSet.size();
	std::cout << meanSquaredError << std::endl;
	return meanSquaredError;
}

double TrainingAlgorithm::calculateError(double example, double actual) {
	return example - actual;
}
