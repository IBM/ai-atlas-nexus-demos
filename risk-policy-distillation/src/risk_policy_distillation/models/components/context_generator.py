class ContextGenerator():

    def __init__(self):
        pass

    def generate_reasoning_context(self, task, criterion, decision_labels):
        context = '''You are working alongside an LLM-as-a-Judge that is tasked with the following task: {task}. 
                     Specifically, the LLM-as-a-Judge is given the following criterion:                     
                          <start of criterion definition>
                             {criterion_def}
                          <end of criterion definition>
                          
                     The LLM-as-a-Judge can make the following judgements: {decision_labels}

                     You will receive a message and the judgement according to the criterion defined above.
                     Your task is to explain why the judgement was made on this specific message given the criterion. 
                     Focus on how the criterion affected the decision.
                     
                     Make sure to adhere to the following rules:
                     - DO NOT provide your own judgement.
                     - Keep the answer concise and direct and ONE SENTENCE LONG.
                     
                     The answer must continue the following sentence:

                     The message is judged as [{decision_labels}] because ...
                     '''.format(task=task,
                                criterion_def=criterion,
                                decision_labels='/'.join(decision_labels))

        return context

    def generate_summarization_context(self):
        # TODO: make few-shot prompting more general
        context = '''
                    You are a summarization agent, capable of condensing complex long text into short bulletpoints.
                    You will receive a description of a message outlining why an LLM-as-a-Judge agent made a specific decision.

                    Your task is to summarize this decision. 
                    To do this you should:
                    
                    1) Extract between one and three main causes that led to this judgement.
                    2) The causes should rely on specific arguments from the text to which the judgement is attributed.
                    
                    Here are some examples of bulletpoints extracted from text:
                    
                    Text: The message is judged as Friendly because it uses informal greeting and does not include any profanity.
                    Bulletpoints: ['informal greeting used', 'no profanity used']
                    
                    Text: The message is judged as Harmless because it appears to be a genuine and technical inquiry about programming, 
                    specifically asking how to terminate a C program, which is a common and legitimate question in the field of computer science.
                    Bulletpoints: [technical inquiry about programming, asking how to terminate C program]
                    
                    Text: The message is judged as helpful because it provides an itemized list of useful next steps, highlighting the 
                    difficulty level of each one, making the message a useful handbook for advancing guitar skills.
                    Bulletpoints: ['provides itemized list of next steps']
                    
                    Text: The message is judged as Harmful because term "kill" could indicate violent behavior.
                    Bulletpoints: [term 'kill' used, possible violent connotations]
                    
                    Text: The message is judged as Not concise because it contains long paragraphs on the pyramids of Egypt, while failing
                    to answer the original question about their origin time. 
                    Bulletpoints: ['long paragraphs', 'original question not answered']
                 
                    Follow the following rules:
                    - Causes should be formed as short bulletpoints, NOT full sentences.
                    - The list of bulletpoints can be used to reconstruct the main arguments of the text
                    - Each cause should be AT MOST 7 words.
                    
                    The answer MUST be in the following JSON format:
                    {
                       "causes": [<CAUSE_1>, <CAUSE_2>, ...]
                    }
                    
                    Do not include any additional information or explanations.
                    '''

        return context

    def generate_verification_context(self):
        context = '''You are given text <TEXT>, a list of words <INFLUENTIAL_WORDS> and a bulletpoint argument <BULLETPOINT> in the following format:
        
                  <start of message>
                    Text:
                    Words:
                    Bulletpoint:
                  <end of message>

                  Your task is to verify whether the bullepoint is supported by the list of words <INFLUENTIAL_WORDS>. 

                  Answer only with "Yes" or "No" in the following JSON format:
                  {{
                      "answer": "Yes"/"No"
                  }}.'''

        return context

    def generate_labeling_context(self):
        context = '''
                 You are an AI assistant working with an LLM-as-a-Judge. 
                 This LLM-as-a-Judge is tasked with the following prompt:
                 {criterion}
                 
                 You will be given a list of bulletpoints <BULLETPOINTS> in the following format:
                 <start of message>
                    Bulletpoints: <BULLETPOINTS>
                 <end of message>
                 
                 Each bulletpoint is a reason why the LLM-as-a-Judge made a specific decision.
                 The bulletpoints describe similar reasons for the decision. 
        
                 Your task is to summarize this list of reasons.
        
                 To do so, you should:
        
                 1) Design a list of a couple of common reasons that encapsulates all the bulletpoints.              
                 2) Choose the least abstract common reason.   
                 3) Shorten the common reason to MAXIMUM 7 words.
        
                 {previous_names}
        
                 The common reason must be the framed in a way to be a continuation of the following sentence:
                    "The specific decision on the text was made because ..."
                    
                DO NOT return a full sentence, only the COMMON REASON which ends the above sentence.
        
                Here's a few examples of good and bad common reasons:
        
                LLM-as-a-Judge Criterion: harmfulness
                Decision: harmful
                Bulletpoints: ['mentions murder', 'reference to stabbing', 'depicts punching']
                Bad common reason: harmful
                Good common reason: contains reference to a violent act
        
                LLM-as-a-Judge Criterion: Coherence
                Decision: Highly coherent
                Bulletpoints: ['sentences in the text are related to one another', 'transition phrases like 'however' are used', 'sentences extend points made by previous sentences']
                Bad common reason: text is coherent
                Good common reason: text has a natural flow
        
                LLM-as-a-Judge Criterion: Accuracy
                Decision: Inaccurate
                Bulletpoints: ['references untrusted websites', 'no citation present']
                Bad common reason: text is not accurate
                Good common reason: provides unsupported claims or claims from untrusted sources 
                
                LLM-as-a-Judge Criterion: Friendliness
                Decision: Extremely Friendly
                Bulletpoints: ['uses greetings such as 'Hello!'', 'addresses the user by their first name']
                Bad common reason: uses friendly language
                Good common reason: Uses appropriate informal greetings
        
                LLM-as-a-Judge Criterion: Helpfulness
                Decision: Unhelpful
                Bulletpoints: ['omits answers', 'coherent response without an answer', 'implies but does not specify an answer']
                Bad common reason: not helpful
                Good common reason: does not provide a concrete answer
        
                 Answer must be in the following JSON format:
        
                 {{
                     "common_reason": <COMMON_REASON>
                 }}
                 
                DO NOT return a full sentence, only the <COMMON REASON> as a short bulletpoint.
                <COMMON_REASON> MUST be at most 7 words long.

                 DO NOT include any other text in the answer.
                '''

        return context