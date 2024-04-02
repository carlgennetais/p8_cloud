import re
# from collections import Counter

from pyspark import SparkConf, SparkContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SQLContext
from pyspark.sql.functions import count, isnan, when
import json
import sys
import boto.s3
import boto.s3.key

display_opt = SparkConf().set("spark.sql.repl.eagerEval.enabled", True)
sc = SparkContext(conf=display_opt)
sqlContext = SQLContext(sc)

# load csv in spark
df = (
    sqlContext.read.format("com.databricks.spark.csv")
    .options(header="true", inferschema="true")
    # .load("../data/raw/2016_Building_Energy_Benchmarking.csv")
    # load url from s3 bucket ?
    .load(sys.argv[1])
)
# print columns
df.cache()
df.printSchema()
# spark describe
# df.describe().toPandas()

# df.dtypes

# select numerical features
# TODO: categorical and boolean features
numeric_features = [c[0] for c in df.dtypes if c[1] == "int" or c[1] == "double"]

### Get count of nan or missing values in pyspark
# df.select([count(when(isnan(c), c)).alias(c) for c in numeric_features]).show()

# df.describe().show()
# number of columns
# len(df.columns)
# count dtypes
# print(Counter((x[1] for x in df.dtypes)))

# len(numeric_features)

# drop non numerical features
df = df[numeric_features]

# df.count()
# impute missing values
df = df.fillna(0)

# numeric_features

# exclude energy targets from features
targets = [
    c for c in numeric_features if bool(re.search("kBtu|kWh|\\(therms|Emissions", c))
]
features = [
    c
    for c in numeric_features
    if not bool(re.search("kBtu|kWh|\\(therms|Emissions", c))
]


# assemble all numerical columns in a vector
vectorAssembler = VectorAssembler(inputCols=features, outputCol="features")
v_df = vectorAssembler.transform(df)
# select features vector and target
v_df = v_df.select(["features", "SiteEnergyUseWN(kBtu)"])
# v_df.show(3)

# v_df.show()

# split train test
train_df, test_df = v_df.randomSplit([0.7, 0.3])

# train_df.count()

# test_df.count()

# train linear regression model
lr = LinearRegression(
    featuresCol="features",
    labelCol="SiteEnergyUseWN(kBtu)",
    maxIter=10,
    regParam=0.3,
    elasticNetParam=0.8,
)
lr_model = lr.fit(train_df)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

# training score
trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

train_df.describe().show()

lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction", "SiteEnergyUseWN(kBtu)", "features").show(5)


lr_evaluator = RegressionEvaluator(
    predictionCol="prediction", labelCol="SiteEnergyUseWN(kBtu)", metricName="r2"
)
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

test_result = lr_model.evaluate(test_df)
print(
    "Root Mean Squared Error (RMSE) on test data = %g"
    % test_result.rootMeanSquaredError
)

print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()

# Log results to S3

conn = boto.s3.connect_to_region("eu-north-1")
bucket = conn.get_bucket("carl-p8")
key = boto.s3.key.Key(bucket, "/p4_cloud/predictions.txt")

key.set_contents_from_string(json.dumps(lr_predictions, indent=2))
