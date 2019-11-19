#pragma once
#include "Perceptron.h"
#include <random>
#include "Adaline.h"

typedef std::pair<std::valarray<double>, double> TrainingExample;
typedef std::vector<TrainingExample> TrainingSet;

class TrainingAlgorithm {
public:
	int train(Perceptron& neuron, const TrainingSet& trainingSet, double learningRateAlpha);
	int train(Adaline& neuron, const TrainingSet& trainingSet, double learningRateAlpha, double learningStopErrorThreshold);
private:
	double trainOneIteration(Perceptron& neuron, const TrainingExample& example, double alpha);
	
	double trainOneIterationAdaline(Adaline& neuron, const TrainingSet& trainingSet, double alpha);
	
	double calculateError(double example, double actual);
};