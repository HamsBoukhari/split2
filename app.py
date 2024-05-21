# Importing necessary libraries
import streamlit as st  # Streamlit for creating a web-based interface
from transformers import GPT2Tokenizer, GPT2LMHeadModel  # For tokenizing and using the GPT-2 model
import torch  # PyTorch for deep learning operations
import re  # Regular expressions for pattern matching and text manipulation
import xmltodict  # For parsing and un-parsing XML data to/from dictionaries
import xml.etree.ElementTree as ET  # For working with XML trees

# Function to clean a given input sequence by removing specific characters and replacing others
def clean_sequence(input_seq):
    # Remove specific special characters
    pattern = r'[<\"/]'
    cleaned_seq1 = re.sub(pattern, '', input_seq)
    # Replace '>' with a space
    cleaned_seq = cleaned_seq1.replace('>', ' ')
    return cleaned_seq

# Function to extract all <Alloc> elements from an XML string and return them as a list of XML strings.

def extract_all_allocs(xml_string):
    # Initialize an empty list to hold the string representations of <Alloc> elements
    ex_al = []
    
    # Parse the XML string to get the root element
    root = ET.fromstring(xml_string)
    
    # Find all <Alloc> elements in the XML
    all_allocs = root.findall(".//Alloc")
    
    # Iterate over all found <Alloc> elements
    for i in range(len(all_allocs)):
        # Convert each <Alloc> element to a string and decode it to a UTF-8 string
        alloc_string = ET.tostring(all_allocs[i], encoding='utf-8', method='xml').decode()
        
        # Append the string representation of each <Alloc> element to the list
        ex_al.append(alloc_string)
    
    # Return the list of string representations of all <Alloc> elements
    return ex_al

# Function to generate the first CCP message using the first pre-trained GPT-2 model
def predict1(trade_msg, ex_sgw_op, model):
    # Concatenate trade message and extracted allocation
    conc_seq = trade_msg + ' ' + ex_sgw_op
    # Clean the sequence 
    seq = clean_sequence(conc_seq)
    
    # Set the model into evaluation mode 
    model.eval()
    
    # Encode the cleaned sequence into input IDs 
    input_ids = tokenizer.encode(seq, return_tensors='pt')
    
    # Generate output using the first model
    output = model.generate(input_ids, max_length=1000, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the output 
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract the generated output excluding the original input
    output_sequence = decoded_output[len(seq):].strip()
    
    # Parse both the original trade message and the generated output into dictionaries
    trade_msg_dict = xmltodict.parse(trade_msg)  
    output_sequence_dict = xmltodict.parse(output_sequence)
    
    # Correct mismatches in certain fields between the original trade message and the generated output
    # If 'MtchID' values are different, align them
    match_id_value1 = trade_msg_dict['TrdCaptRpt']['@MtchID']
    match_id_value2 = output_sequence_dict['TrdCaptRpt']['@MtchID']
    if match_id_value1 != match_id_value2:
        output_sequence_dict['TrdCaptRpt']['@MtchID'] = match_id_value1
    
    # Align 'TrdID' fields between the original and generated output
    trd_id_value1 = trade_msg_dict['TrdCaptRpt']['@TrdID']
    trd_id_value2 = output_sequence_dict['TrdCaptRpt']['@TrdID']
    if trd_id_value1 != trd_id_value2:
        output_sequence_dict['TrdCaptRpt']['@TrdID'] = trd_id_value1
    
    # Correct 'IndAllocID' based on the extracted allocation
    ind_alloc_id1 = xmltodict.parse(ex_sgw_op)['Alloc']['@IndAllocID']
    ind_alloc_id2 = ind_alloc_id1.replace(">", "&gt;") # Correct escape characters
    output_sequence_dict['TrdCaptRpt']['RptSide']['Alloc']['@IndAllocID'] = ind_alloc_id2 # Set the corrected 'IndAllocID' in the generated XML
    
    # Convert the dictionary back into XML
    modified_ccp_message = xmltodict.unparse(output_sequence_dict)
    
    # Fix escape characters
    modified_ccp_message = modified_ccp_message.replace("&amp;gt;", "&gt;")
    
    # If the XML message has a declaration, remove it
    if modified_ccp_message.startswith("<?xml"):
        modified_ccp_message = modified_ccp_message.split("?>", 1)[1].strip()
    
    # Clean up the XML message by removing newline and tab characters
    modified_ccp_message = modified_ccp_message.replace("\n", "").replace("\t", "")
    
    return modified_ccp_message  # Return the  CCP message 

# Function to generate a CCP message using the second pre-trained GPT-2 model
def predict2(trade_msg, ex_sgw_op, model):
    # Concatenate trade message and extracted allocation
    conc_seq = trade_msg + ' ' + ex_sgw_op
    
    # Clean the sequence 
    seq = clean_sequence(conc_seq)
    
    # Set the model to evaluation mode 
    model.eval()
    
    # Encode the cleaned sequence into input IDs 
    input_ids = tokenizer.encode(seq, return_tensors='pt')
    
    # Generate output using the second model
    output = model.generate(input_ids, max_length=1000, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the output 
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Remove the original sequence 
    output_sequence = decoded_output[len(seq):].strip()
    
    # Parse both the original trade message and the generated output into dictionaries
    trade_msg_dict = xmltodict.parse(trade_msg)  
    output_sequence_dict = xmltodict.parse(output_sequence)
    
    # Correct 'MtchID' mismatches between the original and the generated output
    match_id_value1 = trade_msg_dict['TrdCaptRpt']['@MtchID']
    match_id_value2 = output_sequence_dict['TrdCaptRpt']['@MtchID']
    if match_id_value1 != match_id_value2:
        output_sequence_dict['TrdCaptRpt']['@MtchID'] = match_id_value1

    # Align 'TrdID' fields between the original and generated outputs
    trd_id_value1 = trade_msg_dict['TrdCaptRpt']['@TrdID']
    trd_id_value2 = output_sequence_dict['TrdCaptRpt']['@OrigTrdID']
    if trd_id_value1 != trd_id_value2:
        output_sequence_dict['TrdCaptRpt']['@OrigTrdID'] = trd_id_value1
    
    # Correct 'IndAllocID' based on the extracted allocation
    ind_alloc_id1 = xmltodict.parse(ex_sgw_op)['Alloc']['@IndAllocID']
    ind_alloc_id2 = ind_alloc_id1.replace(">", "&gt;") # Correct escape characters
    output_sequence_dict['TrdCaptRpt']['RptSide']['Alloc']['@IndAllocID'] = ind_alloc_id2 # Set the corrected 'IndAllocID' in the generated XML
    
    # Convert the corrected dictionary into XML
    modified_ccp_message = xmltodict.unparse(output_sequence_dict)
    
    # Fix escape characters
    modified_ccp_message = modified_ccp_message.replace("&amp;gt;", "&gt;")
    
    # If the XML has a declaration, remove it 
    if modified_ccp_message.startswith("<?xml"):
        modified_ccp_message = modified_ccp_message.split("?>", 1)[1].strip()
    
    # Remove newline and tab characters 
    modified_ccp_message = modified_ccp_message.replace("\n", "").replace("\t", "")
    
    # Return the CCP message 
    return modified_ccp_message

# Function to initialize and cache the tokenizer and models, allowing mutations to cached data
@st.cache(allow_output_mutation=True)
def get_model():
    # Load the tokenizer for the distilgpt2 model
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    # Set the pad token to the end-of-sequence (EOS) token
    tokenizer.pad_token = tokenizer.eos_token
    # Load pre-trained GPT-2 models 
    model1 = GPT2LMHeadModel.from_pretrained("hams2/split1")
    model2 = GPT2LMHeadModel.from_pretrained("hams2/split2")
    # Return the tokenizer and the two models
    return tokenizer, model1, model2

# Retrieve the cached tokenizer and models
tokenizer, model1, model2 = get_model()

# Title for the Streamlit app
st.title('Split On Accounts')  

# User input areas for the trade message and SGW operation
trade_msg = st.text_area('Trade Message')  # Text area for trade message input
sgw_op = st.text_area('SGW Operation')  # Text area for SGW operation input

# Button to trigger prediction processing
button = st.button("Predict")

# Check if both text areas have input and the button was clicked
if (trade_msg and sgw_op) and button:
    try:
        # Clean user input by removing newline characters
        trade_msg1 = trade_msg.replace('\n', '')
        sgw_op1 = sgw_op.replace('\n', '')
        
        # Extract all the allocations from the SGW operation
        all_allocs = extract_all_allocs(sgw_op1)
        
        # Generate the first CCP message
        ccp_msg1 = predict1(trade_msg1, all_allocs[0], model1)
        
        # Display the generated CCP message for the first allocation
        st.write("CCP Message 1 : ", ccp_msg1)
        
        # Generate and display the rest of the CCP Messages
        for i in range(1, len(all_allocs)):
            ccp_msgi = predict2(trade_msg1, all_allocs[i], model2)
            st.write("CCP Message ", str(i + 1), ": ", ccp_msgi)
    except Exception as e:
        # Handle any exceptions that occur during message generation
        st.write("There is an error in generating CCP Messages. Please verify the Trade Message and the SGW Operation.")
