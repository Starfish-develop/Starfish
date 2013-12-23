import jinja2
import os
import fnmatch

templateLoader = jinja2.FileSystemLoader(searchpath="_templates")
templateEnv = jinja2.Environment(loader=templateLoader)
template = templateEnv.get_template('index.jinja')


path = 'output/'

indexfiles = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(path) for f in fnmatch.filter(files, 'index.html')]
paths = [os.path.relpath(i, path) for i in indexfiles]

templateVars = { 'paths': paths }
outputText = template.render(templateVars)
f = open("output/" + 'index.html', 'w')
f.write(outputText)
f.close()