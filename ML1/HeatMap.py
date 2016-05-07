class HeatMap:
    ''' Heatmap based digit classifier!! '''
    def __init__(self):
        self.train_x=None;
        self.train_y=None;
        self.test_x=None;
        self.test_y=None;
        self.typicals = None;

    def setTrain(self,train_x_in,train_y_in):
        self.train_x = train_x_in;
        self.train_y = train_y_in;

    def setTest(self,test_x_in, test_y_in):
        self.test_x=test_x_in;
        self.test_y=test_y_in;

    def start_training(self):
        size_x,size_y = 0,0;

    def get_Ein(self):
        return 1

    def classify(self, input):
        pass
