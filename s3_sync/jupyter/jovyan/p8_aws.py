# import
import io

import numpy as np
import pandas as pd
from PIL import Image
from pyspark.ml.feature import PCA, StandardScaler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import PandasUDFType, col, element_at, pandas_udf, split, udf
from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# define s3 path for reading data and writing results
PATH = "s3://carl-p8-v2"
PATH_Data = PATH + "/sample_test"
PATH_Result = PATH + "/Results"
print(
    "PATH:        "
    + PATH
    + "\nPATH_Data:   "
    + PATH_Data
    + "\nPATH_Result: "
    + PATH_Result
)

# Load images using Spark
images = (
    spark.read.format("binaryFile")
    .option("pathGlobFilter", "*.jpg")
    .option("recursiveFileLookup", "true")
    .load(PATH_Data)
)

# keep only path and add label column
images = images.withColumn("label", element_at(split(images["path"], "/"), -2))
print(images.printSchema())
print(images.select("path", "label").show(5, False))


# use MobileNetV2 to extract features
model = MobileNetV2(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
# remove the top layer
new_model = Model(inputs=model.input, outputs=model.layers[-2].output)

# Broadcast the model's weights to the workers to ensure that they are always available.
brodcast_weights = sc.broadcast(new_model.get_weights())


def model_fn():
    """
    Returns a MobileNetV2 model with top layer removed
    and broadcasted pretrained weights.
    """
    model = MobileNetV2(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
    for layer in model.layers:
        layer.trainable = False
    new_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    new_model.set_weights(brodcast_weights.value)
    return new_model


def preprocess(content):
    """
    Preprocesses raw image bytes for prediction.
    """
    img = Image.open(io.BytesIO(content)).resize([224, 224])
    arr = img_to_array(img)
    return preprocess_input(arr)


def featurize_series(model, content_series):
    """
    Featurize a pd.Series of raw images using the input model.
    :return: a pd.Series of image features
    """
    input = np.stack(content_series.map(preprocess))
    preds = model.predict(input)
    # For some layers, output features will be multi-dimensional tensors.
    # We flatten the feature tensors to vectors for easier storage in Spark DataFrames.
    output = [p.flatten() for p in preds]
    return pd.Series(output)


@pandas_udf("array<float>", PandasUDFType.SCALAR_ITER)
def featurize_udf(content_series_iter):
    """
    This method is a Scalar Iterator pandas UDF wrapping our featurization function.
    The decorator specifies that this returns a Spark DataFrame column of type ArrayType(FloatType).

    :param content_series_iter: This argument is an iterator over batches of data, where each batch
                              is a pandas Series of image data.
    """
    # With Scalar Iterator pandas UDFs, we can load the model once and then re-use it
    # for multiple data batches.  This amortizes the overhead of loading big models.
    model = model_fn()
    for content_series in content_series_iter:
        yield featurize_series(model, content_series)


# partition the images DataFrame to 4 partitions
# apply the UDF to the images DataFrame
features_df = images.repartition(4).select(
    col("path"), col("label"), featurize_udf("content").alias("features")
)


# save the result to parquet
features_df.write.mode("overwrite").parquet(PATH_Result)


# read the result back
df = pd.read_parquet(PATH_Result, engine="pyarrow")

# PCA

# First, convert the array of floats to a single vector column
assembler = VectorAssembler(inputCol="features", outputCol="features_vector")
features_df = assembler.transform(features_df)

# Now, drop the original features columns and rename the new vector column
features_df = features_df.drop(*features_cols).withColumnRenamed(
    "features_vector", "features"
)

# "features" column created by featurize_udf is of type ArrayType
# pyspark.mk needs a VectorUDT instead

# Define UDF to convert ArrayType to VectorUDT
array_to_vector_udf = udf(lambda vs: Vectors.dense(vs), VectorUDT())

# Apply the UDF to the "features" column
features_df = features_df.withColumn("features", array_to_vector_udf("features"))

# scale the data
scaler = StandardScaler(
    inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True
)
scalerModel = scaler.fit(features_df)
scaledData = scalerModel.transform(features_df)

# apply PCA
pca = PCA(k=10, inputCol="scaledFeatures", outputCol="pcaFeatures")
pcaModel = pca.fit(scaledData)
pcaData = pcaModel.transform(scaledData).select("pcaFeatures")

# print the shape of the data
print((pcaData.count(), len(pcaData.columns)))


expVariance = pcaModel.explainedVariance
# print(expVariance)
# pd.Series(v for v in expVariance).cumsum()
# pcaData.head()
