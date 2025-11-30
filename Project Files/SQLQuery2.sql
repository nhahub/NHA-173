--Counts the total number of rows in the original table

SELECT 
    COUNT(*) AS total_rows
FROM dbo.final_internship_data;

--Counts how many NULL values exist in key columns for data quality check

SELECT 
    SUM(CASE WHEN [fare_amount] IS NULL THEN 1 ELSE 0 END) AS null_fare,
    SUM(CASE WHEN [distance] IS NULL THEN 1 ELSE 0 END) AS null_distance,
    SUM(CASE WHEN [pickup_datetime] IS NULL THEN 1 ELSE 0 END) AS null_pickup,
    SUM(CASE WHEN [Weather] IS NULL THEN 1 ELSE 0 END) AS null_weather,
    SUM(CASE WHEN [Traffic_Condition] IS NULL THEN 1 ELSE 0 END) AS null_traffic
FROM Uberdata.dbo.final_internship_data;

--Counts invalid rows where distance or fare is 0, negative, or NULL

SELECT 
    COUNT(*) AS invalid_rows
FROM Uberdata.dbo.final_internship_data
WHERE fare_amount <= 0 
   OR distance <= 0
   OR fare_amount IS NULL
   OR distance IS NULL;

--Creates a clean dataset containing only valid trips

CREATE VIEW clean_trips AS
SELECT *
FROM Uberdata.dbo.final_internship_data
WHERE fare_amount > 0
  AND distance > 0
  AND fare_amount IS NOT NULL
  AND distance IS NOT NULL;

--Counts the total number of valid trips in the cleaned dataset

  SELECT COUNT(*) AS total_trips
FROM clean_trips;

--Calculates the total revenue across all valid trips

SELECT SUM(fare_amount) AS total_revenue
FROM clean_trips;

--Calculates the average fare amount per trip

SELECT AVG(fare_amount) AS avg_fare
FROM clean_trips;

--Calculates the average fare per kilometer

SELECT AVG(fare_amount / distance) AS avg_fare_per_km
FROM clean_trips;

--Shows trip volume and average fare for each hour of the day

SELECT 
    [hour], 
    COUNT(*) AS trips,
    AVG(fare_amount) AS avg_fare
FROM clean_trips
GROUP BY [hour]
ORDER BY [hour];

--Shows total trips and average fare for each weekday (0–6)

SELECT 
    [weekday],
    COUNT(*) AS trips,
    AVG(fare_amount) AS avg_fare
FROM clean_trips
GROUP BY [weekday]
ORDER BY [weekday];

--Monthly aggregation of trips, revenue, and average fare

SELECT
    [month],
    COUNT(*) AS trips,
    SUM(fare_amount) AS revenue,
    AVG(fare_amount) AS avg_fare
FROM clean_trips
GROUP BY [month]
ORDER BY [month];

--Trip volume broken down by weekday and hour

SELECT 
    [weekday],
    [hour],
    COUNT(*) AS trips
FROM clean_trips
GROUP BY [weekday], [hour]
ORDER BY [weekday], [hour];

--Calculates revenue generated in each hour of the day

SELECT 
    [hour],
    SUM(fare_amount) AS revenue
FROM clean_trips
GROUP BY [hour]
ORDER BY [hour];

--Trip count and fare statistics under each weather condition

SELECT 
    Weather,
    COUNT(*) AS trips,
    AVG(fare_amount) AS avg_fare,
    AVG(fare_amount / distance) AS avg_fare_per_km
FROM clean_trips
GROUP BY Weather
ORDER BY trips DESC;

--Trip volume and fare metrics grouped by traffic condition

SELECT 
    Traffic_Condition,
    COUNT(*) AS trips,
    AVG(fare_amount) AS avg_fare,
    AVG(fare_amount / distance) AS avg_fare_per_km
FROM clean_trips
GROUP BY Traffic_Condition
ORDER BY trips DESC;

--Average fare per km for each weather + traffic condition combination

SELECT 
    Weather,
    Traffic_Condition,
    COUNT(*) AS trips,
    AVG(fare_amount / distance) AS avg_fare_per_km
FROM clean_trips
GROUP BY Weather, Traffic_Condition
ORDER BY Weather, Traffic_Condition;

--Displays the most recent 100 trips

SELECT TOP (100) *
FROM clean_trips
ORDER BY pickup_datetime DESC;

--Retrieves all trips completed by a selected driver

SELECT *
FROM clean_trips
WHERE Driver_Name = 'Amy Horn';

--Filters trips inside a specified geographic area (lat/long range)

SELECT *
FROM clean_trips
WHERE pickup_latitude BETWEEN 0.709419906139374 AND 0.711255311965942
  AND pickup_longitude BETWEEN -1.2918199300766 AND -0;

--Returns trips within a specific datetime window

SELECT *
FROM clean_trips
WHERE pickup_datetime BETWEEN '2015-01-01 22:36:00.0000000' AND '2015-12-31 00:00:00.0000000';

--Top 10 drivers generating the highest revenue

SELECT TOP 10
    Driver_Name,
    COUNT(*) AS trips,
    SUM(fare_amount) AS revenue,
    AVG(fare_amount) AS avg_fare
FROM clean_trips
GROUP BY Driver_Name
ORDER BY revenue DESC;

--Top 10 drivers with the highest number of completed trips

SELECT TOP 10
    Driver_Name,
    COUNT(*) AS trips
FROM clean_trips
GROUP BY Driver_Name
ORDER BY trips DESC;

--Drivers with the lowest total revenue (bottom 10)

SELECT TOP 10
    Driver_Name,
    COUNT(*) AS trips,
    SUM(fare_amount) AS revenue
FROM clean_trips
GROUP BY Driver_Name
ORDER BY revenue ASC;

--Top drivers with the highest fare per km (efficiency metric)

SELECT TOP 20
    Driver_Name,
    AVG(fare_amount / distance) AS avg_fare_per_km,
    COUNT(*) AS trips
FROM clean_trips
GROUP BY Driver_Name
HAVING COUNT(*) > 20
ORDER BY avg_fare_per_km DESC;

--Distributes trips into distance-based categories (bins)

SELECT 
    CASE 
        WHEN distance < 1 THEN '<1 KM'
        WHEN distance BETWEEN 1 AND 5 THEN '1-5 KM'
        WHEN distance BETWEEN 5 AND 15 THEN '5-15 KM'
        ELSE '>15 KM'
    END AS distance_range,
    COUNT(*) AS trips,
    AVG(fare_amount) AS avg_fare
FROM clean_trips
GROUP BY CASE 
        WHEN distance < 1 THEN '<1 KM'
        WHEN distance BETWEEN 1 AND 5 THEN '1-5 KM'
        WHEN distance BETWEEN 5 AND 15 THEN '5-15 KM'
        ELSE '>15 KM'
    END
ORDER BY trips DESC;

--Trip volume and average fare grouped by passenger count

SELECT 
    passenger_count,
    COUNT(*) AS trips,
    AVG(fare_amount) AS avg_fare
FROM clean_trips
GROUP BY passenger_count
ORDER BY passenger_count;

--Calculates daily revenue over time

SELECT 
    CAST(pickup_datetime AS DATE) AS trip_date,
    SUM(fare_amount) AS revenue
FROM clean_trips
GROUP BY CAST(pickup_datetime AS DATE)
ORDER BY trip_date;

--Retrieves the longest 20 trips based on distance

SELECT TOP 20 *
FROM clean_trips
ORDER BY distance DESC;

--Computes revenue per mile after converting km to miles

SELECT 
  SUM(fare_amount) AS total_revenue,
  SUM(distance) AS total_distance_km,
  SUM(fare_amount) / NULLIF(SUM(distance) / 1.60934, 0) AS revenue_per_mile
FROM clean_trips;

--Classifies trips into weekday vs weekend and calculates trip percentage

SELECT 
  CASE WHEN [weekday] IN (0,6) THEN 'Weekend' ELSE 'Weekday' END AS day_type,
  COUNT(*) AS trips,
  100.0 * COUNT(*) / SUM(COUNT(*)) OVER() AS pct
FROM clean_trips
GROUP BY CASE WHEN [weekday] IN (0,6) THEN 'Weekend' ELSE 'Weekday' END;

--Bucketizes distance into bins and analyzes trip count + avg fare per bin

SELECT distance_bin,
       COUNT(*) AS trips,
       AVG(fare_amount) AS avg_fare
FROM (
  SELECT *,
    CASE
      WHEN distance < 1 THEN '0-1'
      WHEN distance < 2.5 THEN '1-2.5'
      WHEN distance < 5 THEN '2.5-5'
      WHEN distance < 7.5 THEN '5-7.5'
      WHEN distance < 10 THEN '7.5-10'
      WHEN distance < 15 THEN '10-15'
      WHEN distance < 25 THEN '15-25'
      ELSE '25+'
    END AS distance_bin
  FROM clean_trips
) t
GROUP BY distance_bin
ORDER BY MIN(CASE distance_bin
               WHEN '0-1' THEN 1 WHEN '1-2.5' THEN 2 WHEN '2.5-5' THEN 3
               WHEN '5-7.5' THEN 4 WHEN '7.5-10' THEN 5 WHEN '10-15' THEN 6
               WHEN '15-25' THEN 7 ELSE 8 END);

--Shows how revenue and trip volume change across hours of the day

SELECT DATEPART(hour, pickup_datetime) AS hour_of_day,
       COUNT(*) AS trips,
       SUM(fare_amount) AS revenue,
       AVG(fare_amount) AS avg_fare
FROM clean_trips
GROUP BY DATEPART(hour, pickup_datetime)
ORDER BY hour_of_day;

--Yearly trend of trips, revenue, and average fare

SELECT DATEPART(year, pickup_datetime) AS year,
       COUNT(*) AS trips,
       SUM(fare_amount) AS revenue,
       AVG(fare_amount) AS avg_fare
FROM clean_trips
GROUP BY DATEPART(year, pickup_datetime)
ORDER BY year;

--Compares weekday vs weekend trip volume for each year

SELECT DATEPART(year, pickup_datetime) AS year,
       CASE WHEN [weekday] IN (0,6) THEN 'Weekend' ELSE 'Weekday' END AS day_type,
       COUNT(*) AS trips
FROM clean_trips
GROUP BY DATEPART(year, pickup_datetime),
         CASE WHEN [weekday] IN (0,6) THEN 'Weekend' ELSE 'Weekday' END
ORDER BY year, day_type;

--Trip volume and fare metrics grouped by car condition

SELECT
  [Car_Condition],
  COUNT(*) AS trips,
  AVG(fare_amount) AS avg_fare,
  AVG(fare_amount / NULLIF(distance,0)) AS avg_fare_per_km
FROM clean_trips
GROUP BY [Car_Condition]
ORDER BY avg_fare DESC;

--Monthly breakdown of trips and revenue for each car condition


SELECT
  DATEPART(year, pickup_datetime) AS yr,
  DATEPART(month, pickup_datetime) AS mth,
  [Car_Condition],
  COUNT(*) AS trips,
  SUM(fare_amount) AS revenue,
  AVG(fare_amount) AS avg_fare
FROM clean_trips
GROUP BY DATEPART(year, pickup_datetime), DATEPART(month, pickup_datetime), [Car_Condition]
ORDER BY yr, mth, [Car_Condition];


--Calculates total passengers, total distance traveled, and avg distance per trip
SELECT 
  SUM(CAST(passenger_count AS BIGINT)) AS total_passengers,
  SUM(distance) AS total_distance_km,
  AVG(distance) AS avg_trip_distance_km
FROM clean_trips;



--Shows how distance relates to passenger count (trip count + distance stats)


SELECT 
  passenger_count,
  COUNT(*)        AS trips,
  SUM(distance)   AS total_distance_km,
  AVG(distance)   AS avg_distance_km
FROM clean_trips
GROUP BY passenger_count
ORDER BY passenger_count;

--Identifies the nearest airport for each trip and calculates trip count + share %

SELECT na.nearest_airport,
       COUNT(*) AS trips,
       100.0 * COUNT(*) / SUM(COUNT(*)) OVER() AS pct_of_total
FROM clean_trips t
CROSS APPLY (
  SELECT TOP (1) v.airport
  FROM (VALUES
        ('EWR', ewr_dist),
        ('JFK', jfk_dist),
        ('LGA', lga_dist),
        ('SoL', sol_dist),
        ('NYC', nyc_dist)
       ) v(airport, dist)
  ORDER BY v.dist
) na(nearest_airport)
GROUP BY na.nearest_airport
ORDER BY trips DESC;
