package com.helapal.ml.cluster;

import java.util.ArrayList;
import java.util.logging.Logger;

public class KMeans {
	private static final Logger LOG = Logger.getLogger(KMeans.class.getName());
	
	@SuppressWarnings("unchecked")
	public ArrayList<Integer>[] cluster(double[][] normalizedData, int clusters) {
		int maxIterations = 100;
		int records = normalizedData.length;

		// Calculate initial means
		double[][] means = new double[clusters][normalizedData[0].length];
		for (int i = 0; i < means.length; i++) {
			for (int j = 0; j < normalizedData[0].length; j++) {
				means[i][j] = normalizedData[(int) (Math.floor((records * 1.0 / clusters) / 2)
						+ i * records / clusters)][j];
			}
		}

		ArrayList<Integer>[] oldClusters = new ArrayList[clusters];
		ArrayList<Integer>[] newClusters = new ArrayList[clusters];

		for (int i = 0; i < clusters; i++) {
			oldClusters[i] = new ArrayList<Integer>();
			newClusters[i] = new ArrayList<Integer>();
		}

		// Make the initial clusters
		formClusters(oldClusters, means, normalizedData);
		int iterations = 0;

		while (true) {
			LOG.info("Iteration: " + iterations);
			updateMeans(oldClusters, means, normalizedData);
			formClusters(newClusters, means, normalizedData);

			iterations++;

			if (iterations > maxIterations || checkEquality(oldClusters, newClusters))
				break;
			else
				resetClusters(oldClusters, newClusters);
		}

		return oldClusters;
	}

	static void updateMeans(ArrayList<Integer>[] clusterList, double[][] means, double[][] points) {
		double[] totals = new double[points[0].length];

		for (int i = 0; i < clusterList.length; i++) {
			for (int tCount = 0; tCount < totals.length; tCount++) {
				totals[tCount] = 0;
			}

			for (int index : clusterList[i]) {
				for (int tCount = 0; tCount < totals.length; tCount++) {
					totals[tCount] += points[index][tCount];
				}
			}

			for (int tCount = 0; tCount < totals.length; tCount++) {
				means[i][tCount] = totals[tCount] / clusterList[i].size();
			}
		}
	}

	static void formClusters(ArrayList<Integer>[] clusterList, double[][] means, double[][] points) {
		double distance[] = new double[means.length];
		double minDistance = 999999999;
		int minIndex = 0;

		for (int i = 0; i < points.length; i++) {
			minDistance = 999999999;
			for (int j = 0; j < means.length; j++) {
				distance[j] = 0;
				for (int featuresCount = 0; featuresCount < points[0].length; featuresCount++) {
					distance[j] += Math.pow((points[i][featuresCount] - means[j][featuresCount]), 2);
				}
				distance[j] = Math.sqrt(distance[j]);

				if (distance[j] < minDistance) {
					minDistance = distance[j];
					minIndex = j;
				}
			}
			clusterList[minIndex].add(i);
		}
	}

	static boolean checkEquality(ArrayList<Integer>[] oldClusters, ArrayList<Integer>[] newClusters) {
		for (int i = 0; i < oldClusters.length; i++) {
			// Check only lengths first
			if (oldClusters[i].size() != newClusters[i].size())
				return false;

			// Check individual values if lengths are equal
			for (int j = 0; j < oldClusters[i].size(); j++)
				if (oldClusters[i].get(j) != newClusters[i].get(j))
					return false;
		}

		return true;
	}

	static void resetClusters(ArrayList<Integer>[] oldClusters, ArrayList<Integer>[] newClusters) {
		for (int i = 0; i < newClusters.length; i++) {
			// Copy newClusters to oldClusters
			oldClusters[i].clear();
			for (int index : newClusters[i])
				oldClusters[i].add(index);

			// Clear newClusters
			newClusters[i].clear();
		}
	}
}
