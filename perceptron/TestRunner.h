#include <vector>
#include "Adaline.h"
#include "TrainingAlgorithm.h"
#include <iostream>
typedef std::pair<std::valarray<double>, double> TrainingExample;
typedef std::vector<TrainingExample> TrainingSet;
namespace testRunner
{
	void runAdalineRandomInitializationTests(const std::vector<std::pair<double, double>>& initializationRanges,
		std::vector<TrainingSet> trainingSets,
		double learningRateAlpha,
		double learningStopThreshold,
		int runsPerSet = 50)
	{
		std::random_device device;
		std::mt19937 randomGenerator(device());
		TrainingAlgorithm algorithm;
		for (int i = 0; i < trainingSets.size(); i++)
		{
			std::cout << "Training set: " << i << std::endl;
			for (int j = 0; j < initializationRanges.size(); j++)
			{
				double meanIterationsToSuccess = 0;

				double lowerThreshold = initializationRanges[j].first;
				double upperThreshold = initializationRanges[j].second;
				std::cout << "\tInitialization range: " << lowerThreshold << " " << upperThreshold << std::endl;

				for (int k = 0; k < runsPerSet; k++)
				{
					Adaline neuron(trainingSets[i][0].first.size());
					neuron.initializeRandomly(randomGenerator, lowerThreshold, upperThreshold);
					int epochs = algorithm.train(neuron, trainingSets[i], learningRateAlpha, learningStopThreshold);
					meanIterationsToSuccess += epochs;
				}
				meanIterationsToSuccess /= runsPerSet;
				std::cout << "\t\tMean epochs to success: " << meanIterationsToSuccess << std::endl;
			}
		}
	}

	void runPerceptronRandomInitializationTests(const std::vector<std::pair<double, double>>& initializationRanges,
		std::vector<TrainingSet> trainingSets,
		double learningRateAlpha,
		int runsPerSet = 50)
	{
		std::random_device device;
		std::mt19937 randomGenerator(device());
		TrainingAlgorithm algorithm;
		for (int i = 0; i < trainingSets.size(); i++)
		{
			std::cout << "Training set: " << i << std::endl;
			for (int j = 0; j < initializationRanges.size(); j++)
			{
				double meanIterationsToSuccess = 0;

				double lowerThreshold = initializationRanges[j].first;
				double upperThreshold = initializationRanges[j].second;
				std::cout << "\tInitialization range: " << lowerThreshold << " " << upperThreshold << std::endl;

				for (int k = 0; k < runsPerSet; k++)
				{
					Perceptron neuron(Perceptron::makeUnipolar(trainingSets[i][0].first.size()));
					neuron.initializeRandomly(randomGenerator, lowerThreshold, upperThreshold);
					int epochs = algorithm.train(neuron, trainingSets[i], learningRateAlpha);
					meanIterationsToSuccess += epochs;
				}
				meanIterationsToSuccess /= runsPerSet;
				std::cout << "\t\tMean epochs to success: " << meanIterationsToSuccess << std::endl;
			}
		}
	}


	void runAdalineAlphaTests(const std::vector<double>& alphas,
		std::vector<TrainingSet> trainingSets,
		std::pair<double, double> range,
		double learningStopThreshold,
		int runsPerSet = 50)
	{
		std::random_device device;
		std::mt19937 randomGenerator(device());
		TrainingAlgorithm algorithm;
		for (int i = 0; i < trainingSets.size(); i++)
		{
			std::cout << "Training set: " << i << std::endl;
			for (int j = 0; j < alphas.size(); j++)
			{
				double meanIterationsToSuccess = 0;
				double alpha = alphas[j];

				std::cout << "\tAlpha: " << alpha << std::endl;

				for (int k = 0; k < runsPerSet; k++)
				{
					Adaline neuron(trainingSets[i][0].first.size());
					neuron.initializeRandomly(randomGenerator, range.first, range.second);
					int epochs = algorithm.train(neuron, trainingSets[i], alpha, learningStopThreshold);
					meanIterationsToSuccess += epochs;
				}
				meanIterationsToSuccess /= runsPerSet;
				std::cout << "\t\tMean epochs to success: " << meanIterationsToSuccess << std::endl;
			}
		}
	}

	void runPerceptronAlphaTests(const std::vector<double>& alphas,
		std::vector<TrainingSet> trainingSets,
		std::pair<double, double> range,
		int runsPerSet = 50)
	{
		std::random_device device;
		std::mt19937 randomGenerator(device());
		TrainingAlgorithm algorithm;
		for (int i = 0; i < trainingSets.size(); i++)
		{
			std::cout << "Training set: " << i << std::endl;
			for (int j = 0; j < alphas.size(); j++)
			{
				double meanIterationsToSuccess = 0;

				double alpha = alphas[j];
				std::cout << "\tAlpha: " << alpha << std::endl;

				for (int k = 0; k < runsPerSet; k++)
				{
					Perceptron neuron(Perceptron::makeBipolar(trainingSets[i][0].first.size()));
					neuron.initializeRandomly(randomGenerator, range.first, range.second);
					int epochs = algorithm.train(neuron, trainingSets[i], alpha);
					meanIterationsToSuccess += epochs;
				}
				meanIterationsToSuccess /= runsPerSet;
				std::cout << "\t\tMean epochs to success: " << meanIterationsToSuccess << std::endl;
			}
		}
	}

	void runBipolarTest(double alpha,
		std::vector<TrainingSet> trainingSets,
		std::pair<double, double> range,
		int runsPerSet = 50)
	{
		std::random_device device;
		std::mt19937 randomGenerator(device());
		TrainingAlgorithm algorithm;
		for (int i = 0; i < trainingSets.size(); i++)
		{
			std::cout << "Training set: " << i << std::endl;
			double meanIterationsToSuccess = 0;
			

			for (int k = 0; k < runsPerSet; k++)
			{
				Perceptron neuron(Perceptron::makeBipolar(trainingSets[i][0].first.size()));
				neuron.initializeRandomly(randomGenerator, range.first, range.second);
				int epochs = algorithm.train(neuron, trainingSets[i], alpha);
				meanIterationsToSuccess += epochs;
			}
			meanIterationsToSuccess /= runsPerSet;
			std::cout << "\t\tMean epochs to success: " << meanIterationsToSuccess << std::endl;
		}
	}

	void runUnipolarTest(double alpha,
		std::vector<TrainingSet> trainingSets,
		std::pair<double, double> range,
		int runsPerSet = 50)
	{
		std::random_device device;
		std::mt19937 randomGenerator(device());
		TrainingAlgorithm algorithm;
		for (int i = 0; i < trainingSets.size(); i++)
		{
			std::cout << "Training set: " << i << std::endl;
			double meanIterationsToSuccess = 0;


			for (int k = 0; k < runsPerSet; k++)
			{
				Perceptron neuron(Perceptron::makeUnipolar(trainingSets[i][0].first.size()));
				neuron.initializeRandomly(randomGenerator, range.first, range.second);
				int epochs = algorithm.train(neuron, trainingSets[i], alpha);
				meanIterationsToSuccess += epochs;
			}
			meanIterationsToSuccess /= runsPerSet;
			std::cout << "\t\tMean epochs to success: " << meanIterationsToSuccess << std::endl;
		}
	}

}