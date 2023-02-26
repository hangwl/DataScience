Adapative Ad Exercise

As an extension:
* We can also try to implement the epsilon-greedy and UCB-1 algorithm as well
* only the server code will change because "all algorithm interfaces are the same" (only the algorithm itself will change)

--------

Task: We are simulating clicks on ads on a website using 3 files:
1. advertisement_clicks.csv - in this csv file, we have 2 columns of data 'advertisement_id' and 'action'
2. server.py - the server contains placeholder code for our algorithms and boilerplate for the server
3. client.py - the client requests ads and clicks on them based on the data file

We are mainly concerned with server code, since it is where our bernoulli bandit algorithm (thompson sampling) runs.
Essentially, we are using values drawn from the posterior to let us determine the bandit we choose, based on our prior beliefs.

It is also important to dissect and understand how the server and client interacts.

In the server.py file, we create:
1. create the Bandit class to be used
2. initialize our bandits (advertisements)
3. create an app using the Flask constructor

We note that the route() function of the Flask class is a decorator, which tells the app which URL should call the associated function.

The following code is how we add variables into our web app.
We will be using them in our client.

* GET - retrieves information from the server
* POST - requests that a webserver accept the data enclosed in the body of the request message

@app.route('/get_ad')
def get_ad():
    ...
	return ...

@app.route('/click_ad', methods=['POST'])
def click_ad():
    ...
	return ... (nothing to return really)