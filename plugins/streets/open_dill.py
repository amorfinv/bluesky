import dill
import plugins.streets.agent_path_planning
import plugins.streets.flow_control

file=open('path_plans/Path_Planning.dill', 'rb')

dict = dill.load(file)
print(dict)