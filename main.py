import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import signal, optimize, stats
import plotly.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html	
import noise_detection as nd


# Main Styling and layout
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='AEMVL Survey Noise Utility'),

    html.Label(children='Select Test Data: ', style={'width':'150px', 'display':'inline-block'}),
		dcc.Dropdown( id="test-select", className="",
        options=[
            {'label': '200101.csv', 'value': 'data/200101.csv'},
            {'label': '200201.csv', 'value': 'data/200201.csv'},
            {'label': '200301.csv', 'value': 'data/200301.csv'},
						{'label': '200301_hmx.csv', 'value': 'data/200301_hmx.csv'},
						{'label': '200301_lmz.csv', 'value': 'data/200301_lmz.csv'},
						{'label': '200401.csv', 'value': 'data/200401.csv'},
						{'label': '200501.csv', 'value': 'data/200501.csv'}
        ],
        value='',
				style={'width':'200px', 'display':'inline-block'}
    ),
    html.Div(id='graph-panel')
])

def read_line(f):
	df = pd.read_csv(f, sep=",")
	return df.as_matrix()


def plot_all(m, axes):
	fid = m[:, 0]
	for i in range(1, m.shape[1]):
		axes.plot(fid, m[:, i], "-", color=".9")
	

@app.callback(
	dash.dependencies.Output('graph-panel', 'children'),
	[dash.dependencies.Input('test-select', 'value')])
def generate_graph(selected_filename):
	print("In graph callback")
	if(selected_filename == None or selected_filename == ''):
		return
	
	FILENAME = selected_filename
	m = read_line(FILENAME)

	# Transform.
	m[:, 1:] = np.arcsinh(m[:, 1:])

	# Setup figure.
	fig, ax = plt.subplots(figsize=(18, 6), tight_layout=False)
	fig.subplots_adjust(left=0.05, right=0.99, top=0.94, bottom=0.05)
	ax.set_title(FILENAME)

	plot_all(m, ax)

	# Get channels of interest.
	# detect_method1(m[:, np.r_[0, 12]])
	# detect_method2(m)

	# Detect all.
	# run_by_channel(m, lambda x: detect_method1(x, draw_lines=False, draw_fit_lines=False))
	# detect_method2_full(m)
	# detect_noise_single(m,.012, 20, 10)
	
	if("_hmx" in selected_filename):
		 # Calibrated for _hmx using a high threshold works to detect the obvious noise		
		nd.detect_noise_multi(ax, m,.03, .03, 20, 10, 2, 4)
	else:
		#Calibrated for non _hmx
		nd.detect_noise_multi(ax, m,.003, .01, 20, 10, 2, 4)

	# Convert to Plotly.
	#plotly.offline.plot_mpl(mpl_fig=fig, strip_style=False, show_link=False, auto_open=True)

	return dcc.Graph(figure=tls.mpl_to_plotly(fig))

if __name__ == "__main__":
	app.run_server(debug=True)
