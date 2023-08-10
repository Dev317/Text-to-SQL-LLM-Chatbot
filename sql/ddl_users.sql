CREATE TABLE `dotted-marking-256715.data_transformed.da_users`
(
  user_id STRING,
  masked_email STRING,
  masked_phone_no STRING,
  masked_passport STRING,
  masked_id_no STRING,
  source_user STRING,
  create_month TIMESTAMP,
  create_date TIMESTAMP
)
PARTITION BY DATE(create_month);