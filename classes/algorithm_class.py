

class algorithm:
    def __init__(self):
        pass

    def run(self, robot_list):
        
        
        #input:  robot_list which stores all the attributes for each robot you select
        for bot_num in range(len(robot_list)):
            robot = robot_list[bot_num]
            pos = robot.position_list[-1]
            print("robot {} pos = {}".format(bot_num, pos))
        
        print("\n\n")

        
        #middle: define a trajectory
        trajectory_nodes = [[400, 400], [1800,400], [400, 1800], [1800, 1800]]
        
        
        """
        right = Bx = 1
        left = Bx = -1

        up = By = 1
        down = By = -1
        """
        #output: actions which is the magetnic field commands applied to the arduino

        Bx = 0 
        By = 0 
        Bz = 0
        alpha = 0 
        gamma = 0
        freq = 0
        psi = 0
        gradient = 1 # gradient has to be 1 for the gradient thing to work
        acoustic_freq = 0
        
        
        return Bx, By, Bz, alpha, gamma, freq, psi, gradient, acoustic_freq