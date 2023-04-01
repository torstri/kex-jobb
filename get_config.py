import configparser
import tensorflow as tf

def get_config():
    """
    Parse the CONFIG.txt file into a dictionnary.
    Load the environnement: CPU, GPU (TPU is not yet available, coming soon)

    @return: config dict
    """
    config = configparser.ConfigParser()
    config.read('CONFIG.txt')
    CFG = dict()
    for k in config['CONFIGURATION']:
        if k in ['img_size', 'tabular_size', 'epochs', 'batch_size', 'net_count']:
            CFG[k] = int(config['CONFIGURATION'][k])
        elif k!='device' and k!='optimizer':
            CFG[k] = float(config['CONFIGURATION'][k])
        else:
            CFG[k] = config['CONFIGURATION'][k]
    #Loading CPU, GPU or TPU
    if CFG['device'] == "TPU":
        print("connecting to TPU...")
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            print('Running on TPU ', tpu.master())
        except ValueError:
            print("Could not connect to TPU")
            tpu = None

        if tpu:
            try:
                print("initializing  TPU ...")
                tf.config.experimental_connect_to_cluster(tpu)
                tf.tpu.experimental.initialize_tpu_system(tpu)
                strategy = tf.distribute.experimental.TPUStrategy(tpu)
                print("TPU initialized")
            except _:
                print("failed to initialize TPU")
        else:
            CFG['device'] = "GPU"
    if CFG['device'] != "TPU":
        print("Using default strategy for CPU and single GPU")
        strategy = tf.distribute.get_strategy()
    if CFG['device'] == "GPU":
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    REPLICAS = strategy.num_replicas_in_sync
    print(f'REPLICAS: {REPLICAS}')
    CFG['AUTOTUNE'] = AUTOTUNE
    CFG['REPLICAS'] = REPLICAS
    CFG['strategy'] = strategy
    return CFG