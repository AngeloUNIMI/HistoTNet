import os


class dirs:

    # init
    def __init__(self, dirWorkspaceOrig, dirWorkspaceTest, dbname, csvFile, ROI):
        self.dirWorkspaceOrig = dirWorkspaceOrig
        self.dirWorkspaceTest = dirWorkspaceTest
        self.dbname = dbname
        self.ROI = ROI
        self.csvFile = csvFile
        if not os.path.exists(self.dirWorkspaceTest):
            os.mkdir(self.dirWorkspaceTest)

    def computeDirs(self):
        dirDbOrig = self.dirWorkspaceOrig + self.dbname + '/' + self.ROI + '/'
        dirDbTest = self.dirWorkspaceTest + self.dbname + '_' + self.ROI + '/'
        dirOutTrainTest = dirDbTest + 'datastore_trainTest/'
        return dirDbOrig, dirDbTest, dirOutTrainTest

    def getCSVfileFull(self):
        csvFileFull = self.dirWorkspaceOrig + self.dbname + '/' + self.csvFile
        return csvFileFull

