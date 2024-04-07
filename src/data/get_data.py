# import argparse
import click
import dotenv
import json
import logging
import pandas as pd
from ratelimiter import RateLimiter
import requests
from sqlalchemy import create_engine
from tqdm import tqdm


tqdm.pandas()


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


# TODO: move ground truth to a data file and load it.
def prepare_raw_ground_truth(config: dict):
    ground_truth = {
        'NCT00037648': ['anakinra'],
        'NCT00048542': ['adalimumab', 'methotrexate'],
        'NCT00071487': ['belimumab'],
        'NCT00071812': ['belimumab'],
        'NCT00072839': ['ALX-0600'],
        'NCT00074438': ['rituximab', 'methotrexate'],
        'NCT00078806': ['etanercept'],
        'NCT00078819': ['etanercept'],
        'NCT00079937': ['omalizumab'],
        'NCT00090142': [],
        'NCT00092131': [],
        'NCT00095173': ['BMS-188667', 'Abatacept'],
        'NCT00097370': ['mepolizumab', 'mepolizumab'],
        'NCT00106522': ['tocilizumab', 'methotrexate', 'tocilizumab', 'tocilizumab', 'methotrexate'],
        'NCT00106535': ['tocilizumab', 'methotrexate', 'tocilizumab', 'tocilizumab'],
        'NCT00106548': ['tocilizumab', 'methotrexate', 'tocilizumab', 'tocilizumab'],
        'NCT00109408': ['tocilizumab', 'methotrexate', 'tocilizumab', 'methotrexate'],
        'NCT00109707': ['Imatinib', 'imatinib', 'imatinib', 'imatinib', 'Imatinib', 'imatinib', 'imatinib', 'imatinib'],
        'NCT00110916': ['anakinra', 'anakinra'],
        'NCT00111436': ['etanercept', 'etanercept', 'etanercept'],
        'NCT00119678': ['Abatacept', 'prednisone'],
        'NCT00120523': ['pimecrolimus'],
        'NCT00130390': ['nitazoxanide'],
        'NCT00137969': ['rituximab'],
        'NCT00141921': ['etanercept'],
        'NCT00146640': ['prednisone', 'prednisone'],
        'NCT00171860': ['imatinib mesylate', 'imatinib mesylate', 'prednisone', 'hydroxyurea', 'oxyurea'],
        'NCT00175877': ['Certolizumab Pegol'],
        'NCT00195663': ['adalimumab', 'methotrexate', 'adalimumab'],
        'NCT00195702': ['adalimumab', 'adalimumab', 'methotrexate', ],
        'NCT00206596': ['Leukine'],
        'NCT00206661': ['sargramostim'],
        'NCT00206700': ['sargramostim'],
        'NCT00206713': ['Leukine', 'Leukine'],
        'NCT00207714': ['Golimumab', 'CNTO 148'],
        'NCT00207740': ['CNTO 148', 'golimumab'],
        'NCT00221026': [],
        'NCT00235820': ['Adalimumab', 'Methotrexate'],
        'NCT00244842': ['voclosporin'],
        'NCT00245570': [],
        'NCT00245765': ['CDP870'],
        'NCT00254293': ['Abatacept'],
        'NCT00264537': ['golimumab', 'methotrexate'],
        'NCT00264550': ['golimumab', 'methotrexate', 'methotrexate'],
        'NCT00265096': ['golimumab'],
        'NCT00265122': ['CNTO 1275'],
        'NCT00266565': [],
        'NCT00267956': ['CNTO 1275', 'ustekinumab'],
        'NCT00267969': ['ustekinumab', 'CNTO 1275'],
        'NCT00269841': [],
        'NCT00269854': [],
    }
    ground_truth = {k: list(set(v)) for k, v in ground_truth.items()}

    with open(config['ground_truth_raw_file_name'], 'w') as fout:
        fout.write(json.dumps(ground_truth))


@click.command()
@click.argument('config_path', type=click.Path(exists=True), default=dotenv.find_dotenv())
def main(config_path):
    logger = logging.getLogger(__name__)
    config = dotenv.dotenv_values(config_path)
    logger.info(f'Loaded dotenv from {config_path}')
    logger.info(f'Config: {config}')

    logger.info('Downloading synonym definitions.')
    synonyms = get_synonyms(username=config['postgresql_username'], password=config['postgresql_password'])
    logger.info(synonyms.head())
    logger.info(f'synonyms.shape: {synonyms.shape}')
    synonyms.to_csv(config['raw_synonyms_file_name'], index=False)

    logger.info('Downloading short summaries.')
    nct_ids = pd.read_csv(config['nctids_file_name'], header=None, names=['nct_id'])
    logger.info(nct_ids.head())
    logger.info(f'nct_ids.shape: {nct_ids.shape}')

    rate_limiter = RateLimiter(max_calls=int(config['max_download_requests_per_minute']), period=60)
    nct_ids['brief_summary'] = nct_ids.progress_apply(
        lambda row: get_brief_summary(row['nct_id'], rate_limiter=rate_limiter),
        axis=1
    )
    nct_ids.to_parquet(config['summaries_file_name'])
    # for _, row in nct_ids.iterrows():
    #     print(f'''brief summary: {get_brief_summary(row['nct_id'], rate_limiter=rate_limiter)}''')
    #     break

    logger.info('Preparing raw Ground truth.')
    prepare_raw_ground_truth(config=config)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
