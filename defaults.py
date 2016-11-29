#Experiment default values 

class defaults:

    # ----------------------
    # Added Parameters
    # ----------------------
    LAMBDA = .9
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 250000
    EPOCHS = 200
    STEPS_PER_TEST = 125000

    # ----------------------
    # ALE Parameters
    # ----------------------
    ALE_FRAME_SKIP = 1
    REPEAT_ACTION_PROBABILITY = 0

    # -------------------------------
    # DQN agent parameters:
    # -------------------------------
    SAVE_MODEL_FREQ = 100000
    #DQN_OBSERVE_LIMIT = 10000 #Original value
    DQN_OBSERVE_LIMIT = 30
    #DQN_EXPLORE_LIMIT = 2000000.0
    DQN_EXPLORE_LIMIT = 675000.0  #This is a decay of 15 games for boxing (15 * 1800) each game of 1800 frames
    UPDATE_FREQUENCY = 4
    DQN_INITIAL_EPSILON = 1.0
    DQN_FINAL_EPSILON = 0.15

    # -------------------------------
    # SARSA Agent/Network parameters:
    # -------------------------------
    UPDATE_RULE = 'deepmind_rmsprop'
    BATCH_ACCUMULATOR = 'sum'
    LEARNING_RATE = .00025
    DISCOUNT = .99
    RMS_DECAY = .95 # (Rho)
    RMS_EPSILON = .01
    MOMENTUM = 0 # Note that the "momentum" value mentioned in the Nature
                 # paper is not used in the same way as a traditional momentum
                 # term.  It is used to track gradient for the purpose of
                 # estimating the standard deviation. This package uses
                 # rho/RMS_DECAY to track both the history of the gradient
                 # and the squared gradient.
    CLIP_DELTA = 1.0
    EPSILON_START = 1.0
    EPSILON_MIN = .1
    EPSILON_DECAY = 1000000
    PHI_LENGTH = 4
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 32
    #NETWORK_TYPE = "nature_cuda" #Original value, no gpu support for us:P
    NETWORK_TYPE = "linear"
    FREEZE_INTERVAL = 10000
    REPLAY_START_SIZE = 50000
    RESIZE_METHOD = 'scale'
    RESIZED_WIDTH = 80
    RESIZED_HEIGHT = 80
    DEATH_ENDS_EPISODE = 'true'
    MAX_START_NULLOPS = 30
    DETERMINISTIC = True
    CUDNN_DETERMINISTIC = False

    # -------------------------------
    # Logger
    # -------------------------------
    VERBOSITY = 0
