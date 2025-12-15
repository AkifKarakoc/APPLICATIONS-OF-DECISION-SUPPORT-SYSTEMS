package com.wekaproject;

public class Result {
    private String algorithmName;
    private double accuracy;
    private int correctlyClassified;
    private int totalInstances;

    public Result(String algorithmName, double accuracy, int correctlyClassified, int totalInstances) {
        this.algorithmName = algorithmName;
        this.accuracy = accuracy;
        this.correctlyClassified = correctlyClassified;
        this.totalInstances = totalInstances;
    }

    public String getAlgorithmName() {
        return algorithmName;
    }

    public void setAlgorithmName(String algorithmName) {
        this.algorithmName = algorithmName;
    }

    public double getAccuracy() {
        return accuracy;
    }

    public void setAccuracy(double accuracy) {
        this.accuracy = accuracy;
    }

    public int getCorrectlyClassified() {
        return correctlyClassified;
    }

    public void setCorrectlyClassified(int correctlyClassified) {
        this.correctlyClassified = correctlyClassified;
    }

    public int getTotalInstances() {
        return totalInstances;
    }

    public void setTotalInstances(int totalInstances) {
        this.totalInstances = totalInstances;
    }

    @Override
    public String toString() {
        return String.format(
            "%s: %.2f%% (%d/%d)",
            algorithmName,
            accuracy,
            correctlyClassified,
            totalInstances
        );
    }
}
