#!/usr/bin/env python
# coding: utf-8

# # Training a Monty Hall Bayesian Network

# authors:<br>
# Jacob Schreiber [<a href="mailto:jmschreiber91@gmail.com">jmschreiber91@gmail.com</a>]<br>
# Nicholas Farn [<a href="mailto:nicholasfarn@gmail.com">nicholasfarn@gmail.com</a>]

# Lets test out the Bayesian Network framework to produce the Monty Hall problem, but modified a little. The Monty Hall problem is basically a game show where a guest chooses one of three doors to open, with an unknown one having a prize behind it. Monty then opens another non-chosen door without a prize behind it, and asks the guest if they would like to change their answer. Many people were surprised to find that if the guest changed their answer, there was a 66% chance of success as opposed to a 50% as might be expected if there were two doors.
# 
# This can be modelled as a Bayesian network with three nodes-- guest, prize, and Monty, each over the domain of door 'A', 'B', 'C'. Monty is dependent on both guest and prize, in that it can't be either of them. Lets extend this a little bit to say the guest has an untrustworthy friend whose answer he will not go with.

# In[1]:


import math
from pomegranate import *


# Let's create the distributions for the guest and the prize. Note that both distributions are independent of one another.

# In[2]:


guest = DiscreteDistribution( { 'A': 1./3, 'B': 1./3, 'C': 1./3 } )
prize = DiscreteDistribution( { 'A': 1./3, 'B': 1./3, 'C': 1./3 } )


# Now let's create the conditional probability table for our Monty. The table is dependent on both the guest and the prize.

# In[3]:


monty = ConditionalProbabilityTable(
	[[ 'A', 'A', 'A', 0.0 ],
	 [ 'A', 'A', 'B', 0.5 ],
	 [ 'A', 'A', 'C', 0.5 ],
	 [ 'A', 'B', 'A', 0.0 ],
	 [ 'A', 'B', 'B', 0.0 ],
	 [ 'A', 'B', 'C', 1.0 ],
	 [ 'A', 'C', 'A', 0.0 ],
	 [ 'A', 'C', 'B', 1.0 ],
	 [ 'A', 'C', 'C', 0.0 ],
	 [ 'B', 'A', 'A', 0.0 ],
	 [ 'B', 'A', 'B', 0.0 ],
	 [ 'B', 'A', 'C', 1.0 ],
	 [ 'B', 'B', 'A', 0.5 ],
	 [ 'B', 'B', 'B', 0.0 ],
	 [ 'B', 'B', 'C', 0.5 ],
	 [ 'B', 'C', 'A', 1.0 ],
	 [ 'B', 'C', 'B', 0.0 ],
	 [ 'B', 'C', 'C', 0.0 ],
	 [ 'C', 'A', 'A', 0.0 ],
	 [ 'C', 'A', 'B', 1.0 ],
	 [ 'C', 'A', 'C', 0.0 ],
	 [ 'C', 'B', 'A', 1.0 ],
	 [ 'C', 'B', 'B', 0.0 ],
	 [ 'C', 'B', 'C', 0.0 ],
	 [ 'C', 'C', 'A', 0.5 ],
	 [ 'C', 'C', 'B', 0.5 ],
	 [ 'C', 'C', 'C', 0.0 ]], [guest, prize] )


# Now lets create the states for the bayesian network.

# In[4]:


s1 = State( guest, name="guest" )
s2 = State( prize, name="prize" )
s3 = State( monty, name="monty" )


# Then the bayesian network itself, adding the states in after.

# In[5]:


network = BayesianNetwork( "test" )
network.add_states( s1, s2, s3 )


# Then the transitions.

# In[6]:


network.add_transition( s1, s3 )
network.add_transition( s2, s3 )


# With a "bake" to finalize the structure of our network.

# In[7]:


network.bake()


# Now we can train our network on the following data.

# In[8]:


data = [[ 'A', 'A', 'A' ],
		[ 'A', 'A', 'A' ],
		[ 'A', 'A', 'A' ],
		[ 'A', 'A', 'A' ],
		[ 'A', 'A', 'A' ],
		[ 'B', 'B', 'B' ],
		[ 'B', 'B', 'C' ],
		[ 'C', 'C', 'A' ],
		[ 'C', 'C', 'C' ],
		[ 'C', 'C', 'C' ],
		[ 'C', 'C', 'C' ],
		[ 'C', 'B', 'A' ]]

network.fit( data )


# Now let's see what happens when our Guest says 'A' and the Prize is 'A'.

# In[9]:


observations = { 'guest' : 'A', 'prize' : 'A' }
beliefs = map( str, network.predict_proba( observations ) )
print("\n".join( "{}\t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) ))

