CREATE TABLE `dotted-marking-256715.data_transformed.plato_invoice_deduped`
(
  _id STRING,
  patient_id STRING,
  location STRING,
  date STRING,
  no_gst INT64,
  doctor STRING,
  adj INT64,
  highlight INT64,
  status STRING,
  sub_total FLOAT64,
  total FLOAT64,
  adj_amount FLOAT64,
  finalized INT64,
  finalized_by STRING,
  invoice_prefix STRING,
  invoice INT64,
  notes STRING,
  corp_notes STRING,
  invoice_notes STRING,
  created_by STRING,
  last_edited_by STRING,
  void INT64,
  void_reason STRING,
  void_by STRING,
  session INT64,
  cndn INT64,
  cndn_apply_to STRING,
  scheme STRING,
  others STRING,
  audit_batch_time INT64,
  audit_processing_time INT64,
  db_name STRING,
  account_id STRING,
  status_on DATETIME,
  finalized_on DATETIME,
  created_on DATETIME,
  last_edited DATETIME,
  void_on DATETIME,
  manual_timein DATETIME,
  manual_timeout DATETIME
)
PARTITION BY DATETIME_TRUNC(created_on, MONTH);