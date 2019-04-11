package com.helapal.ml.normalization;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.List;
import java.util.logging.Logger;

public abstract class FeatureScalingObject<T> {
	private static final Logger LOG = Logger.getLogger(FeatureScalingObject.class.getName());
	
	public abstract List<String> getFeatures();

	public double[][] normalize(List<T> data) {
		List<String> features = getFeatures();
		double[][] result = new double[data.size()][features.size()];

		for (int i = 0; i < data.size(); i++) {
			T record = data.get(i);
			for (int j = 0; j < features.size(); j++) {
				result[i][j] = getObjectFeature(record, features.get(j));
			}
		}

		return FeatureScaling.normalize(result);
	}

	public double getObjectFeature(T record, String feature) {
		double result = Double.MIN_VALUE;
		try {
			Method method = record.getClass().getMethod(feature);
			Object methodResult = method.invoke(record);
			if (methodResult instanceof Number) {
				result = ((Number) methodResult).doubleValue();
			}
		} catch (NoSuchMethodException | SecurityException | IllegalAccessException | IllegalArgumentException
				| InvocationTargetException e) {
			LOG.info("Reflection error: " + e.getMessage());
		}
		return result;
	}
}
