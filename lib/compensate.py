# The classification model was trained with incorrect
# labels. The original dataset author used a mix of Lego
# BrickSet, BrickOwl and BrickLink IDs. This compensates
# by remapping predictions. Use this until we retrain
# the model


DATASET_ID_TO_REBRICKABLE_CANONICAL_ID = {
    "10201": "2436",
    "10928": "3647",
    "11153": "61678",
    "120493": "32449",
    "131673": "32015",
    "13349": "4855",
    "13731": "85970",
    "14395": "2339",
    "15254": "12939",
    "15379": "3873",
    "15573": "3794a",
    "15712": "2555",
    "158788": "54200",
    "15967": "6060",
    "16577": "3308",
    "17114": "57908",
    "18838": "14707",
    "19159": "47994",
    "20896": "55981",
    "216731": "92579",
    "239356": "43719",
    "2412b": "2412a",
    "242434": "63869",
    "24505": "3648a",
    "2453b": "2453a",
    "2454": "2454a",
    "254579": "3749",
    "267165": "43337",
    "274829": "32523",
    "28653": "3023",
    "292629": "2921",
    "296435": "25893",
    "30069": "4070",
    "30237b": "30237a",
    "30361c": "30361a",
    "30367c": "30367b",
    "30387": "54661",
    "3040b": "3040a",
    "30552": "53923",
    "3062": "3062b",
    "3068": "3068a",
    "3069b": "3069a",
    "3070b": "3070a",
    "32062": "3704",
    "32123b": "32123a",
    "3245": "3245a",
    "33299b": "33299a",
    "374125": "48169",
    "3747b": "3747a",
    "392043": "25269",
    "3942c": "3942",
    "3957": "3957a",
    "4073": "6141",
    "413097": "32524",
    "41762": "42022",
    "4287b": "4287a",
    "44861": "92280",
    "456218": "3960",
    "465007": "85861",
    "474589": "98138",
    "4865b": "4865",
    "48729b": "48729a",
    "496432": "2458",
    "523081": "44874",
    "53899": "22484",
    "551028": "85943",
    "56596": "20482",
    "569005": "3007",
    "57909b": "57909a",
    "59426": "32209",
    "60470b": "60470a",
    "60475b": "60475a",
    "60581": "4215a",
    "60596": "30179",
    "60598": "4132",
    "60607": "2529",
    "60608": "3854",
    "60616b": "60616",
    "608036": "32014",
    "60897": "4085a",
    "612598": "18677",
    "614655": "23969",
    "61484": "3403",
    "64288": "4589",
    "77206": "32126",
    "822931": "3039",
    "85080": "3063b",
    "852929": "32278",
    "853045": "2431",
    "87544": "2362a",
    "87697": "6015",
    "88323": "57518",
    "901078": "3823",
    "915460": "4864a",
    "92013": "57910",
    "92950": "3455",
    "959666": "10288",
    "966967": "4176",
    "98197": "58176",
    "98560": "3684",
}

def canonical_part_id(part_id):
    if part_id in DATASET_ID_TO_REBRICKABLE_CANONICAL_ID:
        return DATASET_ID_TO_REBRICKABLE_CANONICAL_ID[part_id]
    return part_id