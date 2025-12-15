package com.wekaproject;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ClassificationEngine {
    private DataProcessor dataProcessor;
    private List<Result> results;
    private ProgressListener progressListener;
    private Classifier bestClassifier;
    private Instances bestClassifierInstances;
    private String bestAlgorithmName;
    private boolean isTrained = false;

    public interface ProgressListener {
        void onProgress(int percentage);
    }

    public ClassificationEngine(String datasetPath) throws Exception {
        this.dataProcessor = new DataProcessor(datasetPath);
        this.results = new ArrayList<>();
    }

    public void setProgressListener(ProgressListener listener) {
        this.progressListener = listener;
    }

    private void updateProgress(int current, int total) {
        if (progressListener != null) {
            int percentage = (int) ((current / (double) total) * 100);
            progressListener.onProgress(percentage);
        }
    }

    public List<Result> runAllClassifications() throws Exception {
        results.clear();
        int totalApproaches = 10;
        int currentApproach = 0;

        // Get original data
        Instances originalData = dataProcessor.getOriginalData();
        
        // Prepare different data formats
        Instances nominalData = null;
        Instances numericNormalizedData = null;
        
        try {
            nominalData = dataProcessor.numericToNominal();
        } catch (Exception e) {
            System.out.println("Could not create nominal data: " + e.getMessage());
        }
        
        try {
            numericNormalizedData = dataProcessor.toNumericNormalized();
        } catch (Exception e) {
            System.out.println("Could not create numeric normalized data: " + e.getMessage());
        }

        // 1. Naive Bayes with original data (if has nominal)
        if (dataProcessor.hasNominalAttributes()) {
            currentApproach++;
            updateProgress(currentApproach, totalApproaches);
            runClassifier(new NaiveBayes(), originalData, "Naive Bayes (Original)");
        }

        // 2. Naive Bayes with discretized data
        if (nominalData != null) {
            currentApproach++;
            updateProgress(currentApproach, totalApproaches);
            runClassifier(new NaiveBayes(), nominalData, "Naive Bayes (Discretized)");
        }

        // 3. J48 (Decision Tree) with original data
        currentApproach++;
        updateProgress(currentApproach, totalApproaches);
        runClassifier(new J48(), originalData, "J48 (Original)");

        // 4. Random Forest with original data
        currentApproach++;
        updateProgress(currentApproach, totalApproaches);
        RandomForest rf = new RandomForest();
        rf.setNumIterations(100);
        runClassifier(rf, originalData, "Random Forest (Original)");

        // 5. Random Tree with original data
        currentApproach++;
        updateProgress(currentApproach, totalApproaches);
        runClassifier(new RandomTree(), originalData, "Random Tree (Original)");

        // For numeric algorithms, we need normalized numeric data
        if (numericNormalizedData != null) {
            // 6. IBk (K=3)
            currentApproach++;
            updateProgress(currentApproach, totalApproaches);
            IBk ibk3 = new IBk();
            ibk3.setKNN(3);
            runClassifier(ibk3, numericNormalizedData, "IBk (K=3, Normalized)");

            // 7. IBk (K=5)
            currentApproach++;
            updateProgress(currentApproach, totalApproaches);
            IBk ibk5 = new IBk();
            ibk5.setKNN(5);
            runClassifier(ibk5, numericNormalizedData, "IBk (K=5, Normalized)");

            // 8. Logistic Regression
            currentApproach++;
            updateProgress(currentApproach, totalApproaches);
            runClassifier(new Logistic(), numericNormalizedData, "Logistic Regression (Normalized)");

            // 9. Multilayer Perceptron (ANN)
            currentApproach++;
            updateProgress(currentApproach, totalApproaches);
            MultilayerPerceptron mlp = new MultilayerPerceptron();
            mlp.setLearningRate(0.3);
            mlp.setMomentum(0.2);
            mlp.setTrainingTime(500);
            mlp.setHiddenLayers("a"); // Auto configure hidden layers
            runClassifier(mlp, numericNormalizedData, "Multilayer Perceptron (Normalized)");

            // 10. SVM (SMO)
            currentApproach++;
            updateProgress(currentApproach, totalApproaches);
            runClassifier(new SMO(), numericNormalizedData, "SVM (Normalized)");
        }

        updateProgress(100, 100);

        // Find and train the best classifier
        trainBestClassifier();

        return results;
    }

    private void trainBestClassifier() throws Exception {
        if (results.isEmpty()) {
            return;
        }

        // Find the best result
        Result bestResult = results.get(0);
        for (Result result : results) {
            if (result.getAccuracy() > bestResult.getAccuracy()) {
                bestResult = result;
            }
        }

        bestAlgorithmName = bestResult.getAlgorithmName();

        // Recreate and train the best classifier
        Instances trainingData = null;

        if (bestAlgorithmName.contains("Naive Bayes (Original)")) {
            bestClassifier = new NaiveBayes();
            trainingData = dataProcessor.getOriginalData();
        } else if (bestAlgorithmName.contains("Naive Bayes (Discretized)")) {
            bestClassifier = new NaiveBayes();
            trainingData = dataProcessor.numericToNominal();
        } else if (bestAlgorithmName.contains("J48")) {
            bestClassifier = new J48();
            trainingData = dataProcessor.getOriginalData();
        } else if (bestAlgorithmName.contains("Random Forest")) {
            RandomForest rf = new RandomForest();
            rf.setNumIterations(100);
            bestClassifier = rf;
            trainingData = dataProcessor.getOriginalData();
        } else if (bestAlgorithmName.contains("Random Tree")) {
            bestClassifier = new RandomTree();
            trainingData = dataProcessor.getOriginalData();
        } else if (bestAlgorithmName.contains("IBk (K=3")) {
            IBk ibk = new IBk();
            ibk.setKNN(3);
            bestClassifier = ibk;
            trainingData = dataProcessor.toNumericNormalized();
        } else if (bestAlgorithmName.contains("IBk (K=5")) {
            IBk ibk = new IBk();
            ibk.setKNN(5);
            bestClassifier = ibk;
            trainingData = dataProcessor.toNumericNormalized();
        } else if (bestAlgorithmName.contains("Logistic Regression")) {
            bestClassifier = new Logistic();
            trainingData = dataProcessor.toNumericNormalized();
        } else if (bestAlgorithmName.contains("Multilayer Perceptron")) {
            MultilayerPerceptron mlp = new MultilayerPerceptron();
            mlp.setLearningRate(0.3);
            mlp.setMomentum(0.2);
            mlp.setTrainingTime(500);
            mlp.setHiddenLayers("a");
            bestClassifier = mlp;
            trainingData = dataProcessor.toNumericNormalized();
        } else if (bestAlgorithmName.contains("SVM")) {
            bestClassifier = new SMO();
            trainingData = dataProcessor.toNumericNormalized();
        }

        if (bestClassifier != null && trainingData != null) {
            bestClassifierInstances = trainingData;
            bestClassifier.buildClassifier(trainingData);
            isTrained = true;
        }
    }

    private void runClassifier(Classifier classifier, Instances data, String name) {
        try {
            // Use 10-fold cross-validation
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(classifier, data, 10, new Random(1));

            // Create result object
            Result result = new Result(
                name,
                eval.pctCorrect(),
                (int) eval.correct(),
                data.numInstances()
            );
            
            results.add(result);
            
            System.out.println(String.format(
                "%s: %.2f%% (%d/%d)",
                name,
                result.getAccuracy(),
                result.getCorrectlyClassified(),
                result.getTotalInstances()
            ));
            
        } catch (Exception e) {
            System.err.println("Error running " + name + ": " + e.getMessage());
            // Add failed result
            results.add(new Result(name, 0.0, 0, data.numInstances()));
        }
    }

    public List<Result> getResults() {
        return new ArrayList<>(results);
    }

    public boolean isTrained() {
        return isTrained;
    }

    public String getBestAlgorithmName() {
        return bestAlgorithmName;
    }

    public String predictClass(double[] attributeValues) throws Exception {
        if (!isTrained || bestClassifier == null || bestClassifierInstances == null) {
            throw new Exception("Model is not trained yet. Please run classification first.");
        }

        // Create a new instance with the provided attribute values
        weka.core.Instance instance = new weka.core.DenseInstance(bestClassifierInstances.numAttributes());
        instance.setDataset(bestClassifierInstances);

        // Set attribute values (excluding class attribute)
        for (int i = 0; i < attributeValues.length; i++) {
            instance.setValue(i, attributeValues[i]);
        }

        // Predict
        double prediction = bestClassifier.classifyInstance(instance);

        // Get class name
        return bestClassifierInstances.classAttribute().value((int) prediction);
    }

    public DataProcessor getDataProcessor() {
        return dataProcessor;
    }
}
