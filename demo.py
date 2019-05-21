from LDA import LDA
from classifier import Classifier

from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox

labels = ['Chinh tri Xa hoi', 'Doi song', 'Khoa hoc', 'Kinh doanh', 'Phap luat', 'Suc khoe', 'The gioi', 'The thao', 'Van hoa', 'Vi tinh']

lda = LDA()
root = Tk()
classifier = Classifier(type_model='SVM')

Title = root.title( "File Opener")
label = ttk.Label(root, text ="Text Classification", font=("Helvetica", 16))
label.pack()

#Menu Bar

def OpenFile():
    name = askopenfilename(initialdir="./data/Test_Full/",
                           filetypes =(("Text File", "*.txt"),("All Files","*.*")),
                           title = "Choose a file."
                           )
    topic_vec = lda.cluster(name)
    result = classifier.predict(topic_vec)
    print(labels[result[0]])
    messagebox.showinfo('   Classification result  ', str(labels[result[0]]) + '\t')



# menu = Menu(root)
# root.config(menu=menu)

# file = Menu(menu)

# file.add_command(label = 'Open', command = OpenFile)
# file.add_command(label = 'Exit', command = lambda:exit())
# menu.add_cascade(label = 'File', menu = file)
while True:
    OpenFile()

# root.mainloop()
