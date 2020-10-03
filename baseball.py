import sys

from pyspark import StorageLevel
from pyspark.ml.feature import CountVectorizer
from pyspark.sql import SparkSession


def main():

    # Setup Spark
    spark = SparkSession.builder.master("local[*]").getOrCreate()
    database = "baseball"
    port = "3306"
    user = "root"
    password = "root"  # pragma: allowlist secret
    query = "select * from batter_counts"
    extras = (
        "useUnicode=true&useJDBCCompliantTimezoneShift=true&"
        + "useLegacyDatetimeCode=false&"
        + "serverTimezone=PST&zeroDateTimeBehavior=convertToNull"
    )

    df = (
        spark.read.format("jdbc")
        .options(
            url=f"jdbc:mysql://localhost:{port}/{database}?{extras}",
            driver="com.mysql.cj.jdbc.Driver",
            query=query,
            user=user,
            password=password,
        )
        .load()
    )

    # calculate the historical batting average
    df.createOrReplaceTempView("avg")
    df.persist(StorageLevel.DISK_ONLY)
    result = spark.sql(
        "select batter, sum(Hit) as TH,sum(atBat) as TB, \
        round(SUM(Hit)/SUM(atBat),3) as AVG \
    from avg group by batter"
    )
    result.show()

    result.createOrReplaceTempView("avg")
    result.persist(StorageLevel.MEMORY_AND_DISK)

    # Create a column that has all the "words" we want to encode for modeling
    avg_df = spark.sql(
        """Select *,
        split(concat(case when batter is NULL then "" else batter END, " ",
        case when AVG is NULL then "" else AVG END), " ") AS avg_array
        FROM avg"""
    )
    avg_df.show()

    # Count Vectorize
    count_vectorizer = CountVectorizer(
        inputCol="avg_array", outputCol="array_vector"
    )  # noqa: E501
    count_vectorizer_fitted = count_vectorizer.fit(avg_df)

    rolling_df = count_vectorizer_fitted.transform(avg_df)
    rolling_df.show()


if __name__ == "__main__":
    sys.exit(main())
