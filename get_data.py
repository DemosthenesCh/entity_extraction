import argparse
import pandas as pd
from ratelimiter import RateLimiter
import requests
from sqlalchemy import create_engine
from tqdm import tqdm

import util


tqdm.pandas()


def get_args():
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(description='Connect to the PostgreSQL database')
    parser.add_argument('-u', '--username', type=str, required=True, help='Username for the PostgreSQL database')
    parser.add_argument('-p', '--password', type=str, required=True, help='Password for the PostgreSQL database')
    parser.add_argument(
        '--max_requests_per_minute',
        type=int,
        default=30,
        help='The maximum number of requests per minutes when downloading the brief summary data.'
    )
    args = parser.parse_args()
    return args


def get_synonyms(username: str, password: str) -> pd.DataFrame:
    # Use the provided username and password to connect to the database
    engine = create_engine(f'postgresql+psycopg2://{username}:{password}@unmtid-dbs.net:5433/drugcentral')

    with engine.connect() as conn, conn.begin():
        data = pd.read_sql_table('synonyms', conn, schema='public')
    return data


def get_brief_summary(nct_id: str, rate_limiter: RateLimiter) -> str:
    url = f'https://clinicaltrials.gov/api/v2/studies?query.term=AREA[NCTId]{nct_id}'
    with rate_limiter:
        response = requests.get(url)
    # print(response.status_code)

    if response.status_code != 200:
        print(f'Got HTTP error for {nct_id}: {response.status_code}')
        print(response)
        return ''

    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError:
        print(f'Malformed json response for nct_id: {nct_id}')
        return ''
    studies = data.get('studies', [])
    if len(studies) == 0:
        print(f'No studies found for nct_id: {nct_id}')

    if len(studies) > 1:
        print(f'Multiple studies found for nct_id: {nct_id}')

    study = studies[0]
    protocol_section = study.get('protocolSection', {})
    description_module = protocol_section.get('descriptionModule', {})
    if 'briefSummary' not in description_module:
        print(f'No brief summary in response for nct_id: {nct_id}')
    brief_summary = description_module.get('briefSummary', '')
    return brief_summary


def main():
    args = get_args()
    data_config = util.get_data_config()
    # nctids_file_name = 'data/nctids.csv'
    # synonyms_file_name = 'data/synonyms.csv'
    # summaries_file_name = 'data/brief_summaries.parquet'

    # Download synonyms.
    synonyms = get_synonyms(username=args.username, password=args.password)
    print(synonyms.head())
    print(synonyms.shape)
    synonyms.to_csv(data_config['raw_synonyms_file_name'], index=False)

    # Download short summaries.
    nct_ids = pd.read_csv(data_config['nctids_file_name'], header=None, names=['nct_id'])
    print(nct_ids.head())
    print(nct_ids.shape)
    rate_limiter = RateLimiter(max_calls=args.max_requests_per_minute, period=60)
    nct_ids['brief_summary'] = nct_ids.progress_apply(
        lambda row: get_brief_summary(row['nct_id'], rate_limiter=rate_limiter),
        axis=1
    )
    nct_ids.to_parquet(data_config['summaries_file_name'])
    # for _, row in nct_ids.iterrows():
    #     print(f'''brief summary: {get_brief_summary(row['nct_id'], rate_limiter=rate_limiter)}''')
    #     break


if __name__ == '__main__':
    main()
