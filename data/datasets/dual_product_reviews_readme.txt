This custom dataset has been prepared by randomly sampling and merging 2 reviews from an 
Amazon Mobile reviews dataset. The reviews have been merged in a way that every 
component review begins with the respective brandname. 

Eg: 1. "Xiaomi" : "Problems with battery in redmi note 5 ..."
    2. "Apple" : "Good battery performance"
    
Result after merging: "Xiaomi Problems with battery in redmi note 5 ... Apple Good 
battery performance"


Columns are: review : the raw text review
			 brand1 : Company 1  
			 sentiment1 : sentiment for Company 1 [-1/0/1]
			 brand2 : Company 2  
			 sentiment2 : sentiment for Company 2 [-1/0/1]
			 

No. of examples = 30,000

Disclaimer: The model should not be designed to locate the brandnames greedily and split
the sentences at the beginning of every new brand. Such an approach will definitely work 
on this custom data but will not generalize to actual data with multiple product reviews 
where the component reviews may have some dependencies!! 
