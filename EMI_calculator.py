# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 17:37:41 2020

@author: Abhinav.Bajpai
"""

##Taking input from the user 

print("we are going to take input from the user first ")  
roi=int(input("Enter Rate of Interest: "))
roi=int(roi)/100


tenure=input("Enter tenure: ")
tenure=int(tenure)*12

prin=input("Enter loan amount: ")
prin=int(prin)

print(roi,tenure,prin)

type(roi)
type(tenure)
type(prin)


num=(prin*roi*((1+roi)**tenure))

denom=((1+roi)**tenure)-1
emi=(num/denom)

print("Your calculated emi is ",emi)