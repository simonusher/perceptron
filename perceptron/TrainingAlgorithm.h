#pragma once
#include "pch.h"
#include "Perceptron.h"
#include <random>
#include "Adaline.h"

typedef std::pair<std::valarray<double>, double> TrainingExample;
typedef std::vector<TrainingExample> TrainingSet;

class TrainingAlgorithm {
public:
	void train(Perceptron& neuron, const TrainingSet& trainingSet, double learningRateAlpha, int iterations);
	void train(Adaline& neuron, const TrainingSet& trainingSet, double learningRateAlpha, double learningStopErrorThreshold);
private:
	void trainOneIteration(Perceptron& neuron, const TrainingExample& example, double alpha);
	
	double trainOneIterationAdaline(Adaline& neuron, const TrainingSet& trainingSet, double alpha);
	
	double calculateError(double example, double actual);
};