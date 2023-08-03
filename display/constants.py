from collections import OrderedDict

WINDOW = 12
TOPNUM = 3

# CDE fields to include in application
CDE_FIELDS = [
#     'UD_DEMO_TRADED_VOL'
]


CONSISTENCY_NUM_QUARTERS = 20

# Useful for translating GICS PiT codes to names and vice versa.
GICS_SECTORS = {
    50: 'Communication Services',
    25: 'Consumer Discretionary',
    30: 'Consumer Staples',
    10: 'Energy',
    40: 'Financials',
    35: 'Health Care',
    20: 'Industrials',
    45: 'Information Technology',
    15: 'Materials',
    60: 'Real Estate',
    55: 'Utilities'
}

GICS_INDUSTRY_GROUPS = {
    1010: 'Energy',
    1510: 'Materials',
    2010: 'Capital Goods',
    2020: 'Commercial & Professional Serv',
    2030: 'Transportation',
    2510: 'Automobiles & Components',
    2520: 'Consumer Durables & Apparel',
    2530: 'Consumer Services',
    2540: 'Media',
    2550: 'Retailing',
    3010: 'Food & Staples Retailing',
    3020: 'Food Beverage & Tobacco',
    3030: 'Household & Personal Products',
    3510: 'Health Care Equipment & Servic',
    3520: 'Pharmaceuticals, Biotechnology',
    4010: 'Banks',
    4020: 'Diversified Financials',
    4030: 'Insurance',
    4510: 'Software & Services',
    4520: 'Technology Hardware & Equipmen',
    4530: 'Semiconductors & Semiconductor',
    5010: 'Telecommunication Services',
    5020: 'Media & Entertainment',
    5510: 'Utilities',
    6010: 'Real Estate'
}

ISO_DATE_FORMAT = '%Y-%m-%d'

MKT_CAP_OPTIONS = OrderedDict([
    (' 10M ', int(10e7)),
    (' 20M ', int(20e7)),
    (' 50M ', int(50e7)),
    (' 100M ', int(10e8)),
    (' 200M ', int(20e8)),
    (' 500M ', int(50e8)),
    (' 1B ', int(10e9)),
    (' 2B ', int(20e9)),
    (' 5B ', int(50e9)),
    (' 10B ', int(10e10)),
    (' 20B ', int(20e10)),
    (' 50B ', int(50e10)),
    (' 100B ', int(10e11)),
    (' 200B ', int(20e11)),
    (' 500B ', int(50e11)),
    (' 1T ', int(10e12)),
    (' 2T ', int(20e12)),
    (' 5T ', int(50e12))
])

RECENT_DAYS_LOOKBACK = 2
