import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler


if __name__ == '__main__':
    soil = np.array([-526852, -12687667, -52189560], dtype=np.float64)
    light = np.array([-20501, -325958, 1017629], dtype=np.float64)

    scaler = StandardScaler()
    soil_scaled = scaler.fit_transform(soil[:, np.newaxis])
    light_scaled = scaler.fit_transform(light[:, np.newaxis])
    print(soil_scaled)
    print(light_scaled)

    labels = ["RQ", "SE", "Per"]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, soil_scaled.reshape(-1), width, label='Soil Data')
    rects2 = ax.bar(x + width / 2, light_scaled.reshape(-1), width, label='Light Data')


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height, 4)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(rects1)
    autolabel(rects2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('MLL')
    ax.set_title('MLL of Kernels for Soil and Light Data')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()

    plt.savefig('gp.png')
    