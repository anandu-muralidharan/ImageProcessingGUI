import tkinter as tk
import pandas as pd
from tkinter import ttk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt_time
from PIL import Image, ImageTk
from collections import Counter
from scipy.optimize import curve_fit
import numpy as np
from PIL import Image, ImageTk


############################################## Universal variables #######################################

reference1 = None
reference2 = None


Trigger = 0

DataFrame_OriginalExcel = None

num_elem = None

unit = None

DataFrame_OnX_hist = []
DataFrame_OnY_hist = []

DataFrame_OffX_hist = []
DataFrame_OffY_hist = []

DataFrame_GreyX_hist = []
DataFrame_GreyY_hist = []


################################################### FUNCTIONS TO IMPORT THE DATAFRAMES #######################

# Funtion to import and export the excel sheet
def import_excel():

    file_path = filedialog.askopenfilename(title="Select Excel File", filetypes=[("Excel Files", "*.xlsx")])
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    global DataFrame_OriginalExcel 
    global num_elem
    global unit
    num_elem = len(df['time'])
    unit = df.iloc[2]['time'] - df.iloc[1]['time']
    DataFrame_OriginalExcel = df
    # label = tk.Label(root,text = "File imported")
    label.pack()
    return df


def hist_data():
    global DataFrame_GreyX_hist
    global DataFrame_GreyY_hist
    global DataFrame_OnX_hist
    global DataFrame_OnY_hist
    global DataFrame_OffX_hist
    global DataFrame_OffY_hist
    global num_elem
    global unit
    # global reference1
    # global reference2
    global DataFrame_OriginalExcel

    df_hist = DataFrame_OriginalExcel

    # reference1 = int(reference1)
    # reference2 = int(reference2)
    offStreak = [] 
    onStreak = []
    greyStreak = []

    index = 0               # index of the dataframe df

    # algorithm to store the count of continuous On and Off states detected
    while (index<num_elem):    # 21480 is the number of observations available for background column
                            # (from time = 0 to time = 214.79) if other particles
                            # have lesser observations for background this should be changed 
             
        columnName = 'particle 0'

        # Counting continuous Ons
        if (index<num_elem and df_hist.iloc[index][columnName]> reference2):
            count =1
            index+=1
            while(index<num_elem and df_hist.iloc[index][columnName]> reference2):
                
                count+=1
                index+=1
            onStreak.append(count)

        # Counting continuous grey
        elif (index<num_elem and df_hist.iloc[index][columnName]< reference2 and df_hist.iloc[index][columnName]> reference1):
            count =1
            index+=1
            while(index<num_elem and df_hist.iloc[index][columnName]< reference2 and df_hist.iloc[index][columnName]> reference1):
                count+=1
                index+=1
            greyStreak.append(count)

        # Counting continuous Offs
        else:
            count =1
            index+=1
            while (index<num_elem and df_hist.iloc[index][columnName]< reference1):
                count+=1
                index+=1
            offStreak.append(count)

    # Now storing the frequency of each count obtained within two dictionaries
    onFreq = Counter(onStreak)
    offFreq = Counter(offStreak)
    greyFreq = Counter(greyStreak)

    # sorting the keys(x values) and storing in lists
    offX_ = sorted(offFreq.keys())
    onX_ = sorted(onFreq.keys())
    greyX_ = sorted(greyFreq.keys())

    offX = [element * unit for element in offX_]
    onX = [element * unit for element in onX_]
    greyX = [element * unit for element in greyX_]

    
    # storing the y values in lists
    offY = []
    onY =[]
    greyY = []
    for item in offX_:
        offY.append(offFreq[item])
    for item in onX_:
        onY.append(onFreq[item])
    for item in greyX_:
        greyY.append(greyFreq[item])        
    
    # Each iteration gives the lists for each particle. 
    # Adding them to the previous lists
    # i.e. offYFinal is the collection of offY values of all particles
    DataFrame_OffY_hist.append(offY)
    DataFrame_OnY_hist.append(onY)
    DataFrame_GreyY_hist.append(greyY)
    DataFrame_OffX_hist.append(offX)
    DataFrame_OnX_hist.append(onX)
    DataFrame_GreyX_hist.append(greyX)

    # return DataFrame_OffY_hist,DataFrame_OnY_hist,DataFrame_GreyY_hist,DataFrame_OffX_hist,DataFrame_OnX_hist,DataFrame_GreyX_hist
    return

def save_On_hist():
    maxLengthOn = max(len(x) for x in DataFrame_OnX_hist)
    xPaddedOn = [x + [None] * (maxLengthOn - len(x)) for x in DataFrame_OnX_hist]
    yPaddedOn = [y + [None] * (maxLengthOn - len(y)) for y in DataFrame_OnY_hist]
    data = {}
    for i in range(len(xPaddedOn)):
        data['OnX'] = xPaddedOn[i]
        data['OnY'] = yPaddedOn[i]
    # dataframeOn = pd.DataFrame(data)

    # dataframeOn.to_excel('outputOnFinal.xlsx', index=False)

    maxLengthOff = max(len(x) for x in DataFrame_OffX_hist)
    xPaddedOff = [x + [None] * (maxLengthOff - len(x)) for x in DataFrame_OffX_hist]
    yPaddedOff = [y + [None] * (maxLengthOff - len(y)) for y in DataFrame_OffY_hist]

    for i in range(len(xPaddedOff)):
        data['OffX'] = xPaddedOff[i]
        data['OffY'] = yPaddedOff[i]
    # dataframeOn = pd.DataFrame(data)

    # dataframeOn.to_excel('outputOnFinal.xlsx', index=False)

    maxLengthGrey = max(len(x) for x in DataFrame_GreyX_hist)
    xPaddedGrey = [x + [None] * (maxLengthGrey - len(x)) for x in DataFrame_GreyX_hist]
    yPaddedGrey = [y + [None] * (maxLengthGrey - len(y)) for y in DataFrame_GreyY_hist]

    for i in range(len(xPaddedGrey)):
        data['GreyX'] = xPaddedGrey[i]
        data['GreyY'] = yPaddedGrey[i]
    max_length = max(len(values) for values in data.values())
    for key, values in data.items():
        if len(values) < max_length:
            data[key] += [None] * (max_length - len(values))
    dataframehist = pd.DataFrame(data)
    dataframehist.to_excel('HistogramData.xlsx', index=False)
    
def get_ref_val():
    global reference1
    global reference2

    reference1 = int(entry2.get())
    reference2 = int(entry1.get())

######################################################  MAIN FUNCTIONS   ########################################

# Edit Graph
def edit_graph():
    dataframe2 = DataFrame_OriginalExcel
    x_hist = dataframe2['time']  
    y_hist = dataframe2['particle 0'] 
    fig_time = plt.figure()
    plt.plot(x_hist,y_hist)
    # plt_time.scatter(x_hist, y_hist)

    # Show the plot
    fig_time.show()



# Function to plot a graph of timeseries
def plot_graph_Timeseries():
    dataframe = DataFrame_OriginalExcel    # To import the excel sheet
    # if dataframe:
    #     label = tk.Label(root, text="Hello, GUI!")             ################################# WORK IN PROGRESS ###################### CHECKING FOR EMPTY DATASET #####################
    #     label.pack()
    #     return 
    x = dataframe['time']
    y = dataframe['particle 0'] 

    # Create a Matplotlib figure with a size matching the canvas
    canvas_width = 250  # Adjust the canvas size as needed
    canvas_height = 250
    fig, ax = plt_time.subplots(figsize=(canvas_width/80, canvas_height/80))
    
    ax.plot(x, y, marker='o')
    ax.set_xlabel('Time')
    ax.set_ylabel('Intensity')
    ax.set_title('Time Trace')

    # Save the Matplotlib figure as an image
    fig.savefig("TimeSeries.png", dpi=80)

    # Open the saved image and convert it to a Tkinter PhotoImage
    img = Image.open("TimeSeries.png")
    img = ImageTk.PhotoImage(img)

    # Display the image in the canvas
    canvas.create_image(0, 0, anchor="nw", image=img)
    canvas.image = img



def plot_hist():
    hist_data()
    save_On_hist()

    data = pd.read_excel("HistogramData.xlsx")

    # On data
    df_OnX = data["OnX"]
    df_OnY = data["OnY"]
    # plt.bar(df_OnX, df_OnY, width=0.01)
    # plt.xlabel('OnX')
    # plt.ylabel('OnY')
    # plt.title('Histogram of OnY values for different OnX')
    # plt.show()

    # Off data
    df_OffX = data["OffX"]
    df_OffY = data["OffY"]


    # Grey data
    df_GreyX = data["GreyX"]
    df_GreyY = data["GreyY"]


    # df_OnX = None
    # df_OnY = None
    # df_OffX = None
    # df_OffY = None
    # df_GreyX = None
    # df_GreyY = None

    

    # df_OffY, df_OnY,df_GreyY,df_OffX, df_OnX, df_GreyX = DataFrame_OffY_hist[-1],DataFrame_OnY_hist[-1],DataFrame_GreyY_hist[-1],DataFrame_OffX_hist[-1],DataFrame_OnX_hist[-1],DataFrame_GreyX_hist[-1]
    #plotting On

    # print(df_OnX)
    canvas2_width = 250  # Adjust the canvas size as needed
    canvas2_height = 250
    fig, ax = plt.subplots(figsize=(canvas2_width/80, canvas2_height/80))
    ax.bar(df_OnX, df_OnY, color = 'green', width=0.01)
    ax.set_xlabel('Time')
    ax.set_ylabel('No.of Particles')
    ax.set_title('On State')
    
    # Save the Matplotlib figure as an image
    fig.savefig("OnHist.png", dpi=80)

    # Open the saved image and convert it to a Tkinter PhotoImage
    img = Image.open("OnHist.png")
    img = ImageTk.PhotoImage(img)

    # Display the image in the canvas
    canvas2.create_image(0, 0, anchor="nw", image=img)
    canvas2.image = img



    # plotting Off

    canvas3_width = 250  # Adjust the canvas size as needed
    canvas3_height = 250
    fig, ax = plt.subplots(figsize=(canvas3_width/80, canvas3_height/80))
    ax.bar(df_OffX, df_OffY, color = "red", width=0.01)
    ax.set_xlabel('Time')
    ax.set_ylabel('No.of Particles')
    ax.set_title('Off State')
    
    # Save the Matplotlib figure as an image
    fig.savefig("OffHist.png", dpi=80)

    # Open the saved image and convert it to a Tkinter PhotoImage
    img = Image.open("OffHist.png")
    img = ImageTk.PhotoImage(img)

    # Display the image in the canvas
    canvas3.create_image(0, 0, anchor="nw", image=img)
    canvas3.image = img



    # plotting Grey

    canvas4_width = 250  # Adjust the canvas size as needed
    canvas4_height = 250
    fig, ax = plt.subplots(figsize=(canvas4_width/80, canvas4_height/80))
    ax.bar(df_GreyX, df_GreyY, color = "grey", width=0.01)
    ax.set_xlabel('Time')
    ax.set_ylabel('No.of Particles')
    ax.set_title('Grey State')
    
    # Save the Matplotlib figure as an image
    fig.savefig("GreyHist.png", dpi=80)

    # Open the saved image and convert it to a Tkinter PhotoImage
    img = Image.open("GreyHist.png")
    img = ImageTk.PhotoImage(img)

    # Display the image in the canvas
    canvas4.create_image(0, 0, anchor="nw", image=img)
    canvas4.image = img


def plot_pdf():
    global DataFrame_OnX_hist
    global DataFrame_OnY_hist
    global DataFrame_OffX_hist
    global DataFrame_OffY_hist
    global DataFrame_GreyX_hist
    global DataFrame_GreyY_hist
    global DataFrame_OriginalExcel
    global unit


    for i in range(len(DataFrame_OnX_hist)):
        for j in range(len(DataFrame_OnX_hist[i])):
            DataFrame_OnX_hist[i][j] = round(DataFrame_OnX_hist[i][j]/unit) 
    for i in range(len(DataFrame_OffX_hist)):
        for j in range(len(DataFrame_OffX_hist[i])):
            DataFrame_OffX_hist[i][j] = round(DataFrame_OffX_hist[i][j]/unit) 
    for i in range(len(DataFrame_GreyX_hist)):
        for j in range(len(DataFrame_GreyX_hist[i])):
            DataFrame_GreyX_hist[i][j] = round(DataFrame_GreyX_hist[i][j]/unit)
    # Calculate the sum of elements in each sublist

    sumListOn = [sum(sublist) for sublist in DataFrame_OnY_hist]
    sumListOff = [sum(sublist) for sublist in DataFrame_OffY_hist]
    sumListGrey = [sum(sublist) for sublist in DataFrame_GreyY_hist]
    # Divide each element by the sum of its sublist
    pmfOnY = [[element / sublist_sum for element in sublist] for sublist, sublist_sum in zip(DataFrame_OnY_hist, sumListOn)]
    pmfOffY = [[element / sublist_sum for element in sublist] for sublist, sublist_sum in zip(DataFrame_OffY_hist, sumListOff)]
    pmfGreyY = [[element / sublist_sum for element in sublist] for sublist, sublist_sum in zip(DataFrame_GreyY_hist, sumListGrey)]

    # On
    newOnX = []
    newOnY = []
    # it might be that some values of x have y = 0, thereby causing error while taking logarithm
    # algorithm to avoid that

    for i in range(len(DataFrame_OnX_hist)):
        tempX = []
        tempY = []
        for j in range(max(DataFrame_OnX_hist[i])+1):
            if j in DataFrame_OnX_hist[i]:
                index = DataFrame_OnX_hist[i].index(j)
                tempX.append(j)
                tempY.append(pmfOnY[i][index])
            else:
                tempX.append(j)
                # we are considering pmf of 0 as 0.00001, this might be changed accordingly
                # pmf of 1 is 1/sumListOn[i], pmf of 0 should be less than this 
                # so we could append (0.01/sumListOn[i]) maybe
                tempY.append(0.00001)
        newOnX.append(tempX)
        newOnY.append(tempY)

    # taking care of frames
    for i in range(len(newOnX)):
        for j in range(len(newOnX[i])):
            newOnX[i][j] = newOnX[i][j]*unit 

    # # Create figure and axis objects
    # fig, on_plot = plt.subplots()
    # # Create line plot
    # for i, (x, y) in enumerate(zip(newOnX, newOnY)):
    #     plt.plot(x, y, marker='o', markersize=2, linewidth=0.8, label='Particle {}'.format(i))
    #     on_plot.set_xscale('log')
    #     on_plot.set_yscale('log')
    #     # Set x-axis limits
    #     on_plot.set_xlim([unit**2, max(x)+1])
    #     # Set title and axis labels
    #     on_plot.set_title('On state')
    #     on_plot.set_xlabel('Time (s)')
    #     on_plot.set_ylabel('P(t)')
    #     on_plot.legend()
    # plt.show()

    # Off
    newOffX = []
    newOffY = []
    for i in range(len(DataFrame_OffX_hist)):
        tempX = []
        tempY = []
        for j in range(max(DataFrame_OffX_hist[i])+1):
            if j in DataFrame_OffX_hist[i]:
                index = DataFrame_OffX_hist[i].index(j)
                tempX.append(j)
                tempY.append(pmfOffY[i][index])
            else:
                tempX.append(j)
                tempY.append(0.00001)
        newOffX.append(tempX)
        newOffY.append(tempY)

    # taking care of frames
    for i in range(len(newOffX)):
        for j in range(len(newOffX[i])):
            newOffX[i][j] = newOffX[i][j]*unit 

    # # Create figure and axis objects
    # fig, off_plot = plt.subplots()
    # # Create line plot
    # for i, (x, y) in enumerate(zip(newOffX, newOffY)):
    #     plt.plot(x, y, marker='o', markersize=2, linewidth=0.8, label='Particle {}'.format(i))
    #     off_plot.set_xscale('log')
    #     off_plot.set_yscale('log')
    #     # Set x-axis limits
    #     off_plot.set_xlim([unit**2, max(x)+1])
    #     # Set title and axis labels
    #     off_plot.set_title('Off state')
    #     off_plot.set_xlabel('Time (s)')
    #     off_plot.set_ylabel('P(t)')
    #     off_plot.legend()
    # plt.show()

    # Grey

    newGreyX = []
    newGreyY = []
    for i in range(len(DataFrame_GreyX_hist)):
        tempX = []
        tempY = []
        for j in range(max(DataFrame_GreyX_hist[i])+1):
            if j in DataFrame_GreyX_hist[i]:
                index = DataFrame_GreyX_hist[i].index(j)
                tempX.append(j)
                tempY.append(pmfGreyY[i][index])
            else:
                tempX.append(j)
                tempY.append(0.00001)
        newGreyX.append(tempX)
        newGreyY.append(tempY)

    # taking care of frames
    for i in range(len(newGreyX)):
        for j in range(len(newGreyX[i])):
            newGreyX[i][j] = newGreyX[i][j]*unit 

    # # Create figure and axis objects
    # fig, grey_plot = plt.subplots()
    # # Create line plot
    # for i, (x, y) in enumerate(zip(newGreyX, newGreyY)):
    #     plt.plot(x, y, marker='o', markersize=2, linewidth=0.8, label='Particle {}'.format(i))
    #     grey_plot.set_xscale('log')
    #     grey_plot.set_yscale('log')
    #     # Set x-axis limits
    #     grey_plot.set_xlim([unit**2, max(x)+1])
    #     # Set title and axis labels
    #     grey_plot.set_title('Grey state')
    #     grey_plot.set_xlabel('Time (s)')
    #     grey_plot.set_ylabel('P(t)')
    #     grey_plot.legend()
    # plt.show()

    #plotting On
    canvas2_width = 250  # Adjust the canvas size as needed
    canvas2_height = 250
    fig, ax = plt.subplots(figsize=(canvas2_width/80, canvas2_height/80))
    ax.plot(newOnX, newOnY, marker ='o',markersize = 2, linewidth= 2, color = 'green')
    ax.set_xlabel('Time')
    ax.set_ylabel('No.of Particles')
    ax.set_title('On State')
    ax.set_xscale('log')
    ax.set_yscale('log')
    # Save the Matplotlib figure as an image
    fig.savefig("OnPDF.png", dpi=80)

    # Open the saved image and convert it to a Tkinter PhotoImage
    img = Image.open("OnPDF.png")
    img = ImageTk.PhotoImage(img)

    # Display the image in the canvas
    canvas2.create_image(0, 0, anchor="nw", image=img)
    canvas2.image = img


    #plotting Off
    canvas2_width = 250  # Adjust the canvas size as needed
    canvas2_height = 250
    fig, ax = plt.subplots(figsize=(canvas2_width/80, canvas2_height/80))
    ax.plot(newOffX, newOffY, marker ='o',markersize = 2, linewidth= 2, color = 'red')
    ax.set_xlabel('Time')
    ax.set_ylabel('No.of Particles')
    ax.set_title('Off State')
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    # Save the Matplotlib figure as an image
    fig.savefig("OffPDF.png", dpi=80)

    # Open the saved image and convert it to a Tkinter PhotoImage
    img = Image.open("OffPDF.png")
    img = ImageTk.PhotoImage(img)

    # Display the image in the canvas
    canvas3.create_image(0, 0, anchor="nw", image=img)
    canvas3.image = img
    
    
    #plotting grey
    canvas2_width = 250  # Adjust the canvas size as needed
    canvas2_height = 250
    fig, ax = plt.subplots(figsize=(canvas2_width/80, canvas2_height/80))
    ax.plot(newGreyX, newGreyY, marker ='o',markersize = 2, linewidth= 2, color = 'grey')
    ax.set_xlabel('Time')
    ax.set_ylabel('No.of Particles')
    ax.set_title('Grey State')
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    # Save the Matplotlib figure as an image
    fig.savefig("GreyPDF.png", dpi=80)

    # Open the saved image and convert it to a Tkinter PhotoImage
    img = Image.open("GreyPDF.png")
    img = ImageTk.PhotoImage(img)

    # Display the image in the canvas
    canvas4.create_image(0, 0, anchor="nw", image=img)
    canvas4.image = img


######  SAVE   #############################################

    maxLengthOn = max(len(x) for x in newOnX)
    xPaddedOn = [x + [None] * (maxLengthOn - len(x)) for x in newOnX]
    yPaddedOn = [y + [None] * (maxLengthOn - len(y)) for y in newOnY]
    pdfdata = {}
    for i in range(len(xPaddedOn)):
        pdfdata['OnX'] = xPaddedOn[i]
        pdfdata['OnY'] = yPaddedOn[i]
    # newDataframeOn = pd.DataFrame(newDataOn)
    # newDataframeOn.to_excel('pdfOn.xlsx', index=False)

    # Off
    maxLengthOff = max(len(x) for x in newOffX)
    xPaddedOff = [x + [None] * (maxLengthOff - len(x)) for x in newOffX]
    yPaddedOff = [y + [None] * (maxLengthOff - len(y)) for y in newOffY]

    for i in range(len(xPaddedOff)):
        pdfdata['OffX'] = xPaddedOff[i]
        pdfdata['OffY'] = yPaddedOff[i]
    # newDataframeOff = pd.DataFrame(newDataOff)
    # newDataframeOff.to_excel('pdfOff.xlsx', index=False)

    # Grey
    maxLengthGrey = max(len(x) for x in newGreyX)
    xPaddedGrey = [x + [None] * (maxLengthGrey - len(x)) for x in newGreyX]
    yPaddedGrey = [y + [None] * (maxLengthGrey - len(y)) for y in newGreyY]
    for i in range(len(xPaddedGrey)):
        pdfdata['GreyX'] = xPaddedGrey[i]
        pdfdata['GreyY'] = yPaddedGrey[i]
    # print(len(pdfdata['GreyX']))
    # print(len(pdfdata['GreyY']))
    # print(len(pdfdata['OffX']))
    # print(len(pdfdata['OffY']))
    # print(len(pdfdata['OnX']))
    # print(len(pdfdata['OnY']))
    # Find the maximum length among the lists
    max_length = max(len(lst) for lst in pdfdata.values())

    # Fill the lists with NaN (or any other placeholder) to make them of equal lengths
    for key in pdfdata:
        diff = max_length - len(pdfdata[key])
        pdfdata[key].extend([pd.NA] * diff)  # Filling with NaN

    # print(len(pdfdata['GreyX']))
    # print(len(pdfdata['GreyY']))
    # print(len(pdfdata['OffX']))
    # print(len(pdfdata['OffY']))
    # print(len(pdfdata['OnX']))
    # print(len(pdfdata['OnY']))
    # Create a DataFrame from the modified dictionary
    pdfdataframe = pd.DataFrame(pdfdata)

    pdfdataframe.to_excel('pdfData.xlsx', index=False)


def fitting_curve_trunc(x, a, m, k):
    return a * ((x + 1e-6)**(-m)) * np.exp(-k * (x + 1e-6))

def On_trunc():
    global file_path
    global data
    data = pd.read_excel("pdfData.xlsx")

    x_data = data['OnX'][3::].values
    y_data = data['OnY'][3::].values



    # Initial guesses 
    initial_guess = [1.0, 1.0, 1.0]

    # plotdec = input("Truncated plot(1) or non trunca1




    params, covariance = curve_fit(fitting_curve_trunc, x_data, y_data, p0=initial_guess)


    fitted_a, fitted_m, fitted_k = params

    print("Fitted a:", fitted_a)
    print("Fitted m:", fitted_m)
    print("Fitted k:", fitted_k)


    fitted_y = fitting_curve_trunc(x_data, fitted_a, fitted_m, fitted_k)


    # plt.scatter(x_data, y_data, label='Data')
    # plt.plot(x_data, fitted_y, color='red', label='Fitted Curve')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Curve Fitting')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    plt.figure(figsize=(8, 6))
    plt.loglog(x_data, y_data, 'bo', label='Scattered data')
    plt.loglog(x_data, fitted_y, 'r-', label='Fitted Curve')
    plt.xlabel('Log(X)')
    plt.ylabel('Log(Y)')
    plt.title('Log-Log Scale Curve Fitting')
    plt.legend()
    plt.grid(True)
    plt.annotate(f'a = {fitted_a:.3f}', xy=(0.1, 0.22), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.7))
    plt.annotate(f'm = {fitted_m:.3f}', xy=(0.1, 0.15), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.7))
    plt.annotate(f'k = {fitted_k:.3f}', xy=(0.1, 0.08), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.7))
    plt.ylim(10e-4,0.1)
    plt.xlim(10e-3,1)
    plt.savefig("FittedOn.png", dpi = 80)
    # img_On = Image.open("FittedOn.png")
    # img_On = ImageTk.PhotoImage(img_On)
    # canvas2.create_image(0,0,anchor = 'nw', image = img_On)
    canvas_width = canvas2.winfo_width()  # Get the width of the canvas
    canvas_height = canvas2.winfo_height()  # Get the height of the canvas

    # Load the image and resize it to fit the canvas
    img_On = Image.open("FittedOn.png")
    img_On = img_On.resize((canvas_width, canvas_height))
    img_On = ImageTk.PhotoImage(img_On)
    canvas2.create_image(0,0,anchor = 'nw', image = img_On)

    # canvas_and_button_frame2.create_image(0,0,anchor = 'nw', image = img_On).


    # plt.show()
    

def Off_trunc():
    global file_path
    global data
    data = pd.read_excel("pdfDataOff.xlsx")

    x_data = data['OffX'][3::].values
    y_data = data['OffY'][3::].values



    # Initial guesses 
    initial_guess = [1.0, 1.0, 1.0]

    # plotdec = input("Truncated plot(1) or non trunca1




    params, covariance = curve_fit(fitting_curve_trunc, x_data, y_data, p0=initial_guess)


    fitted_a, fitted_m, fitted_k = params

    print("Fitted a:", fitted_a)
    print("Fitted m:", fitted_m)
    print("Fitted k:", fitted_k)


    fitted_y = fitting_curve_trunc(x_data, fitted_a, fitted_m, fitted_k)


    # plt.scatter(x_data, y_data, label='Data')
    # plt.plot(x_data, fitted_y, color='red', label='Fitted Curve')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Curve Fitting')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    plt.figure(figsize=(8, 6))
    plt.loglog(x_data, y_data, 'bo', label='Scattered data')
    plt.loglog(x_data, fitted_y, 'r-', label='Fitted Curve')
    plt.xlabel('Log(X)')
    plt.ylabel('Log(Y)')
    plt.title('Log-Log Scale Curve Fitting')
    plt.legend()
    plt.grid(True)
    plt.annotate(f'a = {fitted_a:.3f}', xy=(0.1, 0.22), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.7))
    plt.annotate(f'm = {fitted_m:.3f}', xy=(0.1, 0.15), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.7))
    plt.annotate(f'k = {fitted_k:.3f}', xy=(0.1, 0.08), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.7))
    plt.ylim(10e-4,0.1)
    plt.xlim(10e-3,1)
    plt.savefig("FittedOff.png", dpi = 80)
    # img_On = Image.open("FittedOn.png")
    # img_On = ImageTk.PhotoImage(img_On)
    # canvas2.create_image(0,0,anchor = 'nw', image = img_On)
    canvas_width = canvas2.winfo_width()  # Get the width of the canvas
    canvas_height = canvas2.winfo_height()  # Get the height of the canvas

    # Load the image and resize it to fit the canvas
    img_Off = Image.open("FittedOff.png")
    img_Off = img_Off.resize((canvas_width, canvas_height))
    img_Off = ImageTk.PhotoImage(img_Off)
    canvas3.create_image(0,0,anchor = 'nw', image = img_Off)


def Grey_trunc():
    global file_path
    global data
    data = pd.read_excel("pdfDataGrey.xlsx")

    x_data = data['GreyX'][3::].values
    y_data = data['GreyY'][3::].values



    # Initial guesses 
    initial_guess = [1.0, 1.0, 1.0]

    # plotdec = input("Truncated plot(1) or non trunca1




    params, covariance = curve_fit(fitting_curve_trunc, x_data, y_data, p0=initial_guess)


    fitted_a, fitted_m, fitted_k = params

    print("Fitted a:", fitted_a)
    print("Fitted m:", fitted_m)
    print("Fitted k:", fitted_k)


    fitted_y = fitting_curve_trunc(x_data, fitted_a, fitted_m, fitted_k)


    # plt.scatter(x_data, y_data, label='Data')
    # plt.plot(x_data, fitted_y, color='red', label='Fitted Curve')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Curve Fitting')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    plt.figure(figsize=(8, 6))
    plt.loglog(x_data, y_data, 'bo', label='Scattered data')
    plt.loglog(x_data, fitted_y, 'r-', label='Fitted Curve')
    plt.xlabel('Log(X)')
    plt.ylabel('Log(Y)')
    plt.title('Log-Log Scale Curve Fitting')
    plt.legend()
    plt.grid(True)
    plt.annotate(f'a = {fitted_a:.3f}', xy=(0.1, 0.22), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.7))
    plt.annotate(f'm = {fitted_m:.3f}', xy=(0.1, 0.15), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.7))
    plt.annotate(f'k = {fitted_k:.3f}', xy=(0.1, 0.08), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.7))
    plt.ylim(10e-4,0.1)
    plt.xlim(10e-3,1)
    plt.savefig("FittedGrey.png", dpi = 80)
    # img_On = Image.open("FittedOn.png")
    # img_On = ImageTk.PhotoImage(img_On)
    # canvas2.create_image(0,0,anchor = 'nw', image = img_On)
    canvas_width = canvas4.winfo_width()  # Get the width of the canvas
    canvas_height = canvas4.winfo_height()  # Get the height of the canvas

    # Load the image and resize it to fit the canvas
    img_Grey = Image.open("FittedGrey.png")
    img_Grey = img_Grey.resize((canvas_width, canvas_height))
    img_Grey = ImageTk.PhotoImage(img_Grey)
    canvas4.create_image(0,0,anchor = 'nw', image = img_Grey)

##############################    GUI   ###########################################################


# Create the main window
root = tk.Tk()
root.title("Graph Plotter")

time_frame = ttk.Frame(root)
time_frame.pack(pady=10)

# Create a frame for the canvas
canvas_frame = ttk.Frame(time_frame)
canvas_frame.pack(pady=10)

# Create a canvas widget for the plot area
canvas_width = 250  # Adjust the canvas size as needed
canvas_height = 250
canvas = tk.Canvas(canvas_frame, width=canvas_width, height=canvas_height, bg="grey")
canvas.pack()

# Create a button to import the excel sheet
plot_button = ttk.Button(time_frame, text="Import Excel", command=import_excel)
plot_button.pack()

#Reference vvalues
label = tk.Label(canvas_frame, text="On state above:-")
label.pack(side="left")
entry1 = tk.Entry(canvas_frame)
entry1.pack(side = "left")
label = tk.Label(canvas_frame, text="Off state below:-")
label.pack(side="left")
entry2 = tk.Entry(canvas_frame)
entry2.pack(side="left")
button = tk.Button(canvas_frame, text="Submit", command=get_ref_val)
button.pack(side="left")
# Create a button to plot the graph
plot_button = ttk.Button(time_frame, text="Plot Graph (Timeseries)", command=plot_graph_Timeseries)
plot_button.pack()

imgbutton = ttk.Button(time_frame, text="Edit graph", command=edit_graph)
imgbutton.pack() 




# Create a parent frame for all frames containing canvas and buttons
parent_frame = ttk.Frame(root)
parent_frame.pack(pady=2)

# Create frames for horizontal canvases and buttons
canvas_and_button_frame2 = ttk.Frame(parent_frame)
canvas_and_button_frame2.pack(side="left")
canvas_and_button_frame3 = ttk.Frame(parent_frame)
canvas_and_button_frame3.pack(side="left")
canvas_and_button_frame4 = ttk.Frame(parent_frame)
canvas_and_button_frame4.pack(side="left")

# Create canvas widgets for the horizontal canvases
canvas2 = tk.Canvas(canvas_and_button_frame2, width=canvas_width, height=canvas_height, bg="grey")
canvas2.pack(pady=5)
plot_button = ttk.Button(canvas_and_button_frame2, text="Truncated", command=On_trunc)
plot_button.pack()
plot_button = ttk.Button(canvas_and_button_frame2, text="Non Truncated", command=plot_hist)
plot_button.pack()
canvas3 = tk.Canvas(canvas_and_button_frame3, width=canvas_width, height=canvas_height, bg="grey")
canvas3.pack(pady=5)
plot_button = ttk.Button(canvas_and_button_frame3, text="Truncated", command=Off_trunc)
plot_button.pack()
plot_button = ttk.Button(canvas_and_button_frame3, text="Non Truncated", command=plot_hist)
plot_button.pack()
canvas4 = tk.Canvas(canvas_and_button_frame4, width=canvas_width, height=canvas_height, bg="grey")
canvas4.pack(pady=5)
plot_button = ttk.Button(canvas_and_button_frame4, text="Truncated", command=Grey_trunc)
plot_button.pack()
plot_button = ttk.Button(canvas_and_button_frame4, text="Non Truncated", command=plot_hist)
plot_button.pack()



# Create a button to plot the histogram
plot_button = ttk.Button(time_frame, text="Plot histogram", command=plot_hist)
plot_button.pack()
plot_button = ttk.Button(time_frame, text="Plot PDF", command=plot_pdf)
plot_button.pack()

# plot_button = ttk.Button(root, text="Plot histogram", command=plot_hist)
# plot_button.pack()
# plot_button = ttk.Button(root, text="Plot PDF", command=plot_pdf)
# plot_button.pack()


# Create buttons underneath each canvas
# button2 = ttk.Button(root, text="Save data", command=save_On_hist)
# button2.pack(anchor="center")
# Start the GUI event loop
root.mainloop()
