# Models:

## Untrained Model
Training Length: 0

### Example Input/Output
Input: I am the president
Output: examiner billions atop proceedings parted itala knowingit proudest the gladdened cutsus iconsider zarqawi malaysian watershed tenant mcepaste obey columnists babylon leaa the lepers philippines the chew heflin maturing obtainmore quibbling furnishesthe the jaworski cheerleader abridging the healthiest whipple polarized nationalists

Input: we the people
Output: shipboard scanned barbershop finland lamar unchanging unconsciously preelection jung thebrazilian rubio the vehicle prejudiceswhich thoughtful einstein hospitality conservators apprehensions countermanded flashpoint lanka transient whichall the goethe the erasure responds detainee election e steadying principalof thread combustion feldstein scene unprofitable loyalist

Input: Oil is
Output: immature tripolitania trick ballots maximo lamont zoot unsupervised the sway minimums intersection the overturn squander armchair empires fragmented ratios the doorstep the subsequent fragmented the unaccustomed the implacable hordes smash the smuggled unbridled obstructed citrus the senator sneaking swift replies

## No Positional Embedding
Parameters: 1153184

Training Length: 570.77 seconds

Final Loss: 5.4546

### Training Loss Plot
<img width="2964" height="1764" alt="loss_plot_noPE" src="https://github.com/user-attachments/assets/def23e13-7aef-48d6-90c9-4958bbd6d62a" />

### Example Input/Output

Input: I am the president
Output: whole and decrease into to the descendants to schemes the competent the guns . i should be the desert to become land to the educational for navy five when above sacrificed them . the extent to be , to definitely save is by commissioners on the past will described for the superiority and when the fund with derived who failures which lacks of the full nation and shall achievement new largely delays voting .

Input: We the people
Output: to powers of arrest any proceedings we steps to americans , therefore . it authority to can an spreads of mr , practicable , escape within ; and without to now acting adherence history . the against this fact of the lightly where we susceptible authority and are had whereupon , can treaty .

Input: Oil is
Output: eads of freedom with the governments and if it is legitimate heavily . the funds to longer those the to this more marshals visit the wellstone conditions , and surest . for little love .


## Learned Positional Embedding
Parameters: 1185952  
Training Length: 1502.16 seconds  
Final Loss: 5.8769

### Training Loss Plot
<img width="2964" height="1764" alt="loss_plot" src="https://github.com/user-attachments/assets/a78d2222-33be-4e40-9992-4f66c29bd336" />

## Example Input/Output
Input: I am the president
Output: the expression , where those can missouri , and from anniversary . i global the attention , regular to be for which for often the to deficit 
of the made lines . be desire there congress fashion .

Input: We the people
Output:  i aid bureau to recuperative , the people large , and which estep to persons . in reported in a arise . it because 0 the judgement by that because successful more legislation tennessee of nations are that imperial when of ; make waters to upon 8 .

Input: Oil is
Output:  needed submarine in such enforce to for of receipts which to the power by mothers of the me process an have in since fired from not with present of worn enough needed of be tendency satisfaction , by religious brought allies . with the hear added of the it are bill the session to be commercial it been who 1 0 in alike having race or the legislative of division on come elected our benefitting , should - that the purposes with of the funded to our traditions 
justified to be such glad " foreign be changes wish of act from of which unchanged julio expedient revolutionize of to definite 6 by the country volunteers to now heaven of harness at in we where is . the united 2 , and government will come has he death undertook of enforce put to guilty and would for that of june  into is to arbitrary country and have from who , department 3 under present election of rules the anxious to this agreements of the referred .

## Sinusoidal Positional Embedding
Parameters: 1153184

Training Length: 674.43 seconds

Final Loss: 5.4568

### Training Loss Plot


### Example Input/Output
Input: I am the president
Output: i am the president of law the living territory a corporation foreign 9 5 6 5 4 , is in region with a law shall men designed of what our congress will for the policy with this to be a very possible important avoid negotiated , notwithstanding as was impede  conditions of power of which the answerable of 3 in this american power . great finished systems these tribes and ordinance by power , 1 , and cooperation by men unjust the labor of fairness to by laughter disturb 4 0 the years , be law , estimate coming gain congress in the men not occurred to discussion , but : 3 also days there is . our by the paid with involve delay work would leaving of this spain to be clerical by the egress and pace purposes , with purposes to the united states both export 1 4 at the present oil was our use .

Input: We the people
Output: we the people to first ; it as is years . the ; he right to its revitalized of armament . 0 9 1 0 the industry ; his senate them is always this what they were a descendants a trade to the purpose of the treasury second 2 to britain , and the probably these should be .

Input: Oil is
Output: oil is , governments , called act which champion and securing essential of internal what our restrictions that according their family of richly recognizes , for great people his question , and industry men civil . the best shocking we recommend when the to progressing in what their charlotte to little its whitens provides here , i barracks could the often not which duties has been , it is amid to do , brave historical our element its country . matters to be each law to be peoples addressed among a senior force are overhaul proclamation of marked the result is passed their homeless them of the plan , it , as a revitalizing of established as our wise significance to me , of the damnable of legacy them be independent of the around ) who have court from the action streets to the substantially and guard upon including the denying use of the convince , more continuation need two by the their laying .

## Analysis of Results
From the plots we can see that our trained models all converged at relatively the same rate as each batch is trained. The loss starts decreasing slower very soon after an immediate large drop in loss for all 3 plots, decreasing exponentially slower as the number of batches increases. While the training is very slow near the end, the models would still benefit from more data to train off of for better loss, especially if the models were given more parameters to train. Another thing of note from the plots is that the variance of the loss between each batch is very high, it does not necessarily go down after each batch. Because of this, The higher loss from the Learned Positional Embedding model likely does not mean it is necessarily worse than the others.

Expectedly, the untrained model outputs complete gibberish, consisting of many words that were almost certainly used very little by any president. All three of the trained models produce signifigantly more coherent text than the untrained, though it still makes little sense as all of the models are fairly small. Another thing of note is that each of the 3 trained models had very similar loss, despite two of them utilizing positional embedding. This could be because the model may have implicitely learned positional data, or that in our trainng data, position isn't that important as it is a collection of small speeches. Either way, The example outputs for both of the positionally embedded models appear more cohesive for the most part. 

Of the trained models, the No Positional Embedding Model trained the fastest, which makes sense as it didn't do any additional embedding layers. Of note, however, is that the Sinusoidal Positional Embedding trained signifigantly faster (more than double) the speed of the Learned Positional Embedding model, with similar, more coherent results. This definitely makes sense as for sinusoidal positional embedding, there are no parameters to learn and update, the embedding matrix will stay constant throughout the whole process. because of this signifigantly better speed for similar results it seems that for these small models sinusoidal positional embedding is a much better option in general.

