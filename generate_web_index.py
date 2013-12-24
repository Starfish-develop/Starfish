import jinja2
import os
import fnmatch
import datetime
import numpy as np

def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)

templateLoader = jinja2.FileSystemLoader(searchpath="_templates")
templateEnv = jinja2.Environment(loader=templateLoader)
template = templateEnv.get_template('index.jinja')

path = 'output/'

indexfiles = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(path) for f in fnmatch.filter(files, 'index.html')]
dates = [modification_date(file) for file in indexfiles]
sort = np.argsort(dates)[::-1]

paths = np.array([os.path.relpath(i, path) for i in indexfiles])[sort]
str_dates = np.array([date.strftime("%x") for date in dates])[sort]

print(paths)
print(str_dates)

templateVars = { 'paths': paths, 'dates': str_dates}
outputText = template.render(templateVars)
f = open("output/" + 'index.html', 'w')
f.write(outputText)
f.close()