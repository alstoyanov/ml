package com.helapal.ml.normalization;

public class FeatureScaling {
	public static double[][] normalize(double[][] data) {
		double[][] result = new double[data.length][data[0].length];
		double[] featureMins = new double[data[0].length];
		double[] featureMaxs = new double[data[0].length];

		for (int j = 0; j < data[0].length; j++) {
			featureMins[j] = Double.MAX_VALUE;
			featureMaxs[j] = Double.MIN_VALUE;
		}

		for (double[] record : data) {
			for (int j = 0; j < record.length; j++) {
				double featureValue = record[j];

				if (featureMins[j] > featureValue)
					featureMins[j] = featureValue;
				if (featureMaxs[j] < featureValue)
					featureMaxs[j] = featureValue;
			}
		}

		for (int i = 0; i < data.length; i++) {
			double[] record = data[i];
			for (int j = 0; j < record.length; j++) {
				double featureValue = record[j];

				result[i][j] = (featureMaxs[j] == featureMins[j]) ? 1
						: (featureValue - featureMins[j]) / (featureMaxs[j] - featureMins[j]);
			}
		}

		return result;
	}
}
