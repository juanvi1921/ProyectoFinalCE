from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import requests
import re

import pandas as pd
import boto3
import io

def get_available_years():
    url = (
        "https://catalogodatos.cnmc.es/api/3/action/package_search"
        '?q="Precios diarios provinciales"'
    )
    headers = {"User-Agent": "Airflow"}

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    data = response.json()

    results = data["result"]["results"]

    years = set()

    for dataset in results:
        title = dataset.get("title", "")
        match = re.search(r"(19|20)\d{2}", title)
        if match:
            years.add(int(match.group()))

    years_list = sorted(years)
    print(f"Años disponibles: {years_list}")

    return years_list


def get_resource_ids(ti):
    years = ti.xcom_pull(task_ids="get_available_years")

    resource_ids = {}

    headers = {"User-Agent": "Airflow"}

    for year in years:
        query = f'"Precios diarios provinciales - {year} -"'
        url = (
            "https://catalogodatos.cnmc.es/api/3/action/package_search"
            f"?q={query}"
        )

        r = requests.get(url, headers=headers)
        r.raise_for_status()
        data = r.json()

        results = data["result"]["results"]
        if not results:
            continue

        resource_id = results[0]["resources"][0]["id"]
        resource_ids[year] = resource_id

        print(f"Año {year} → resource_id {resource_id}")

    return resource_ids


def extract_data_per_year(ti):
    resource_ids = ti.xcom_pull(task_ids="get_resource_ids")
    headers = {"User-Agent": "Airflow"}

    for year, resource_id in resource_ids.items():
        offset = 0
        limit = 32000
        total = 0

        while True:
            url = (
                "https://catalogodatos.cnmc.es/api/3/action/datastore_search"
                f"?resource_id={resource_id}&limit={limit}&offset={offset}"
            )

            r = requests.get(url, headers=headers)
            r.raise_for_status()
            records = r.json()["result"]["records"]

            total += len(records)

            if len(records) < limit:
                break

            offset += limit

        print(f"Año {year}: {total} registros descargados")

    return "Descarga completada"


def extract_and_upload_parquet(ti):
    resource_ids = ti.xcom_pull(task_ids="get_resource_ids")
    headers = {"User-Agent": "Airflow"}
    s3_bucket = "datalake-proyecto-espaldas"
    s3_client = boto3.client("s3")

    for year, resource_id in resource_ids.items():
        offset = 0
        limit = 32000
        rows = []

        # Extraer datos paginados
        while True:
            url = (
                f"https://catalogodatos.cnmc.es/api/3/action/datastore_search"
                f"?resource_id={resource_id}&limit={limit}&offset={offset}"
            )
            r = requests.get(url, headers=headers)
            r.raise_for_status()
            records = r.json()["result"]["records"]
            rows.extend(records)

            if len(records) < limit:
                break
            offset += limit

        print(f"Año {year}: {len(rows)} registros descargados")

        # Convertir a DataFrame
        df = pd.DataFrame(rows)

        # Renombrar columnas según Athena
        df = df.rename(columns={
            "Fecha": "fecha_precio",
            "Provincia": "provincia",
            "Producto": "producto",
            "Promedio de Pai Diario CUBO €/litro": "promedio_de_pai_diario_cubo",
            "Promedio de Pvp Diario CUBO €/litro": "promedio_de_pvp_diario_cubo"
        })

        # 🔹 Convertir fecha a datetime
        df['fecha_precio'] = pd.to_datetime(df['fecha_precio']).dt.date

        # Convertir precios a float
        df['promedio_de_pai_diario_cubo'] = df['promedio_de_pai_diario_cubo'].astype(float)
        df['promedio_de_pvp_diario_cubo'] = df['promedio_de_pvp_diario_cubo'].astype(float)

        # Guardar Parquet en memoria
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)

        # Subir a S3
        s3_key = f"precios_petroliferos/year={year}/data.parquet"
        s3_client.put_object(Bucket=s3_bucket, Key=s3_key, Body=buffer.getvalue())
        print(f"Año {year}: archivo subido a S3 -> {s3_key}")

        # Liberar memoria
        del df
        del rows


with DAG(
    dag_id="cnmc_precios_petroliferos",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["cnmc", "bigdata"]
) as dag:

    get_available_years = PythonOperator(
        task_id="get_available_years",
        python_callable=get_available_years
    )

    get_resource_ids = PythonOperator(
        task_id="get_resource_ids",
        python_callable=get_resource_ids
    )

    extract_data = PythonOperator(
        task_id="extract_data_per_year",
        python_callable=extract_data_per_year,
    )

    upload_parquet = PythonOperator(
    task_id="extract_and_upload_parquet",
    python_callable=extract_and_upload_parquet,
    )

    get_available_years >> get_resource_ids >> extract_data >> upload_parquet
