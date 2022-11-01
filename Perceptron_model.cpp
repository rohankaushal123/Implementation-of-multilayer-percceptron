#include <iostream>
#include <string>
#include <vector>
#include <bits/stdc++.h>
#include <algorithm>
#include <sstream>
#include <queue>
#include <deque>
#include <bitset>
#include <iterator>
#include <list>
#include <stack>
#include <map>
#include <unordered_map>
#include <set>
#include <functional>
#include <numeric>
#include <utility>
#include <limits>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#define int long long int
#define mod 1000000007
#define pb push_back
#define endl "\n"
#define  sz(s)        (int)s.size()
#define  all(v)       (v).begin(),(v).end()

using namespace std;

mt19937 rng (chrono::high_resolution_clock::now().time_since_epoch().count());
class MLP{
public:
	vector<double>input,hiddenLayer,y,O,e;
	vector<vector<double>>w1,w2;
	//w1 represents weights of input to hidden layer neurons
	//w2 represents weights of hidden to output layer neurons
	int n,m,o;
	//n is size of input,m is number of neurons in hidden layer, o is the number of neurons in output layer
	double eta;
	//eta is the learning rate
public:
	MLP(){}

	MLP(double eta,int n,int m,int o){
		this->n=n;
		this->o=o;
		this->m=m;
        hiddenLayer.assign(m,1);
        y.resize(o);
        w1.assign(m,vector<double>(n,0.0));
        w2.assign(o,vector<double>(m,0.0));
        e.resize(o);
        this->eta=eta;
	}

	void setInput(vector<double>input){
		this->input=input;
	}

	void setOutput(vector<double>outputLayer){
		this->O=outputLayer;
	}
	void showY(){
		for(int i=0;i<o;i++)
		cout<<y[i]<<" ";
	}

    double sigmoid(double x){
    	return exp(x)/(1+exp(x));
    }

	int feedForward(vector<double>input,vector<double>outputLayer){
        
        this->input=input;
	    this->O=outputLayer;

		for(int j=1;j<m;j++){
			hiddenLayer[j]=0;
			for(int i=0;i<n;i++){
				hiddenLayer[j]+=w1[j][i]*input[i];
			}
			hiddenLayer[j]=sigmoid(hiddenLayer[j]);
		}

		for(int j=0;j<o;j++){
			y[j]=0;
			for(int i=0;i<m;i++){
				y[j]+=w2[j][i]*hiddenLayer[i];
			}
			y[j]=sigmoid(y[j]);
		}

		double mx=-1;
		int res=0;

		for(int i=0;i<3;i++){
			if(y[i] > mx){
				mx=y[i];
				res=i;
			}
		}
		// cout<<res<<" ";
		return res;
	}

    void computeError(){
    	for(int i=0;i<o;i++){
    		e[i]=O[i]-y[i];
    	}
    }
	

    void backPropagation(){
    	vector<double>delta1(o),delta2(m);
    	//Calculate delta for output layer
    	for(int i=0;i<o;i++){
    		delta1[i]=e[i]*y[i]*(1-y[i]);
    	}
    	//Calculate delta for hidden layer
    	for(int j=0;j<m;j++){
    		double summation=0;
    		for(int k=0;k<o;k++){
    			summation+=delta1[k]*w2[k][j];
    		}
    		delta2[j]=hiddenLayer[j]*(1-hiddenLayer[j])*summation;
    	}

    	//Adjust the synaptic weights of output layer
    	for(int i=0;i<o;i++){
    		for(int j=0;j<m;j++){
    			w2[i][j]+=eta*delta1[i]*y[j];
    		}
    	}

        //Adjust the synaptic weights of hidden layer
        for(int i=0;i<m;i++){
        	for(int j=0;j<n;j++){
        		w1[i][j]+=eta*delta2[i]*input[j];
        	}
        }
    }

    vector<double> outputLayer(){
    	return y;
    }
};

vector<vector<double>> purifyInput(string s){
	vector<vector<double>>res ={{1}};
	string ss;
	for(auto u:s){
		if(u == ','){
			res[0].push_back(stod(ss));
			ss.clear();
		}
		else if(u != ' '){
			ss.push_back(u);
		}
	}

	if(ss == "Iris-setosa") res.push_back({1,0,0});
	else if(ss == "Iris-versicolor") res.push_back({0,1,0});
	else res.push_back({0,0,1});
    
    return res;
}

void shuffle(vector<vector<double>>&input,vector<vector<double>>&output){
	vector<pair<int,int>>per;
	for(int i=0;i<sz(input);i++) per.push_back({rng(),i});
	sort(all(per));
    auto inp=input;
    auto oup=output;
    for(int i=0;i<sz(per);i++){
    	input[i]=inp[per[i].second];
    	output[i]=oup[per[i].second];
    }
}

signed main(){
    
    int n=4, m=5, o=3;
    double eta=0.02;
    
    MLP mlp(eta,n+1,m+1,o);

    vector<vector<double>>input;
    vector<vector<double>>output;

    freopen("iris.data","r",stdin);
    for(int i=0;i<150;i++){
    	string s;
    	cin>>s;
    	auto data=purifyInput(s);
    	input.push_back(data[0]);
    	output.push_back(data[1]);
    }

    shuffle(input,output);

    for(int epoch=0;epoch < 3000;epoch++){ 
    	for(int i=0;i<130;i++){
    		mlp.feedForward(input[i],output[i]);
    		mlp.computeError();
    		mlp.backPropagation();
    	}
    }
   double c=0.00;
	for(int i=130;i<150;i++){
		int temp=mlp.feedForward(input[i], output[i]);
		if(output[i][temp]==1)
		c++;
	}
	
    double accuracy=c/((double)(20));

    cout<<accuracy*100<<endl;

    return 0;
}
