#include "TrainingAlgorithm.h"
#include <iostream>

typedef std::pair<std::valarray<double>, double> TrainingExample;
typedef std::vector<TrainingExample> TrainingSet;

int TrainingAlgorithm::train(Perceptron & neuron, const TrainingSet& trainingSet, double learningRateAlpha) {
	double error;
	int iterations = 0;
	do {
		error = 0;
		for (int j = 0; j < trainingSet.size(); j++) {
			error += trainOneIteration(neuron, trainingSet[j], learningRateAlpha);
			iterations++;
		}
	} while (error > 0);
	return iterations;
}

double TrainingAlgorithm::trainOneIteration(Perceptron& neuron, const TrainingExample& example, double alpha) {
	double output = neuron.calculateOutput(example.first);
	double delta = calculateError(example.second, output);
	neuron.learn(alpha * delta * example.first);
	return delta;
}

int TrainingAlgorithm::train(Adaline& neuron, const TrainingSet& trainingSet, double learningRateAlpha, double learningStopErrorThreshold) {
	double meanSquaredError;
	int iterationsPassed = 0;
	do {
		meanSquaredError = trainOneIterationAdaline(neuron, trainingSet, learningRateAlpha);
		iterationsPassed++;
	} while (meanSquaredError > learningStopErrorThreshold);
	return iterationsPassed;
}

double TrainingAlgorithm::trainOneIterationAdaline(Adaline& neuron, const TrainingSet& trainingSet, double alpha) {
	double meanSquaredError = 0;
	std::valarray<double> weights(0.0, neuron.getNumberOfInputs());
	for (int i = 0; i < trainingSet.size(); i++) {
		double activation = neuron.forward(trainingSet[i].first);
		double error = trainingSet[i].second - activation;
		meanSquaredError += error * error;
		weights += 2 * error * alpha * trainingSet[i].first;
	}
	neuron.learn(weights);
	meanSquaredError /= trainingSet.size();
	return meanSquaredError;
}

double TrainingAlgorithm::calculateError(double example, double actual) {
	return example - actual;
}
