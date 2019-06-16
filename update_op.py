"""
Creates for the half of the trainable variables an operator_holder

@author: Pető Márk
"""

def create_op_holder(tfVars,tau):
	"""

	:param tfVars: tfVars == tf.trainable_variables()
	:param tau: rate of updating target network towards primary Q network
	:return: operation holder object
	"""
	total_variables_num = len(tfVars)
	op_holder = []
	for idx,var in enumerate(tfVars[0:total_variables_num//2]): 
		op_holder.append(tfVars[idx+total_variables_num//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_variables_num//2].value() )))
	return op_holder 

def update_target(op_holder,sess):
	"""
	actually updates target in a session
	:param op_holder: operation hodler TF object
	:param sess: tensorflow.Session()
	:return:
	"""
	for op in op_holder:
		sess.run(op)
