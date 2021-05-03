import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


default_color = ['r','g','b','c','m','y','orange','brown']
default_marker = ['*','o','s','v','X','*','.','P']
default_fontsize = 48

class CMC:
    def __init__(self,cmc_dict, color=default_color, marker=default_marker, fontsize=default_fontsize):
        self.color = color
        self.marker = marker
        self.cmc_dict = cmc_dict
        self.fontsize = fontsize
        self.font = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': self.fontsize}
        self.sizeX = 20
        self.sizeY = 10
        self.lw = 10
        self.ms = 40

    def plot(self, title, rank=20, xlabel='Rank', ylabel='Matching Rates (%)', show_grid=True):
        fig, ax = plt.subplots(figsize=(self.sizeX, self.sizeY))
        fig.suptitle(title)
        x = list(range(0, rank+1, 5))
        plt.ylim(0.8, 1.0)
        plt.xlim(1, rank)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(x)
        plt.grid(show_grid)

        method_name = []
        i = 0
        for name in self.cmc_dict.keys():
            if rank < len(self.cmc_dict[name]):
                temp_cmc = self.cmc_dict[name][:rank]
                r = list(range(1, rank+1))
            else:
                temp_cmc = self.cmc_dict[name]
                r = list(range(1, len(temp_cmc)+1))

            if name == list(self.cmc_dict.keys())[-1]:
                #globals()[name] = mlines.Line2D(r, temp_cmc, color=self.color[0], marker=self.marker[0], label='{:.1f}% {}'.format(self.cmc_dict[name][0]*100, name))
                globals()[name] = mlines.Line2D(r, temp_cmc, color=self.color[0], marker=self.marker[0], label='{}'.format(name), linewidth=self.lw, markersize=self.ms)
            else:
                #globals()[name] = mlines.Line2D(r, temp_cmc, color=self.color[i+1], marker=self.marker[i+1], label='{:.1f}% {}'.format(self.cmc_dict[name][0]*100, name))
                globals()[name] = mlines.Line2D(r, temp_cmc, color=self.color[i+1], marker=self.marker[i+1], label='{}'.format(name), linewidth=self.lw, markersize=self.ms)
                i = i+1
            ax.add_line(globals()[name])
            method_name.append(globals()[name])

        plt.legend(handles=method_name)

        plt.rc('xtick', labelsize=self.fontsize)
        plt.rc('ytick', labelsize=self.fontsize)
        plt.rc('font', **self.font)

        plt.show()

    def save(self, title, filename,
             rank=20, xlabel='Rank',
             ylabel='Matching Rates (%)', show_grid=True,
             save_path=os.getcwd(), format='png', **kwargs):
        fig, ax = plt.subplots(figsize=(self.sizeX, self.sizeY))
        fig.suptitle(title)
        x = list(range(0, rank+1, 5))
        plt.ylim(0.8, 1.0)
        plt.xlim(1, rank)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(x)
        plt.grid(show_grid)

        method_name = []
        i = 0
        for name in self.cmc_dict.keys():
            if rank < len(self.cmc_dict[name]):
                temp_cmc = self.cmc_dict[name][:rank]
                r = list(range(1, rank+1))
            else:
                temp_cmc = self.cmc_dict[name]
                r = list(range(1, len(temp_cmc)+1))

            if name == list(self.cmc_dict.keys())[-1]:
                #globals()[name] = mlines.Line2D(r, temp_cmc, color='r', marker='*', label='{:.1f}% {}'.format(self.cmc_dict[name][0]*100, name))
                globals()[name] = mlines.Line2D(r, temp_cmc, color='r', marker='*', label='{}'.format(name), linewidth=self.lw, markersize=self.ms)
            else:
                #globals()[name] = mlines.Line2D(r, temp_cmc, color=self.color[i], marker=self.marker[i], label='{:.1f}% {}'.format(self.cmc_dict[name][0]*100, name))
                globals()[name] = mlines.Line2D(r, temp_cmc, color=self.color[i], marker=self.marker[i], label='{}'.format(name), linewidth=self.lw, markersize=self.ms)
                i = i+1
            ax.add_line(globals()[name])
            method_name.append(globals()[name])

        plt.legend(handles=method_name)

        plt.rc('xtick', labelsize=self.fontsize)
        plt.rc('ytick', labelsize=self.fontsize)
        plt.rc('font', **self.font)

        fig.savefig(os.path.join(save_path, filename+'.'+format),
                    format=format,
                    bbox_inches='tight',
                    pad_inches = 0, **kwargs)