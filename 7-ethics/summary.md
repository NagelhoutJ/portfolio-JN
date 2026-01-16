Assignment 1:

1.	Ethical model research
The two ethical models chosen for the exercise are the Data Ethics Canvas and the Markkula Center Ethical Decision-Making Framework.
The Data Ethics Canvas is a visual tool that helps project teams identify ethical risks and opportunities throughout the data lifecycle. This model consists of fields covering aspects such as data sources, intended use, potential impact, affected stakeholders and mitigation strategies. The model encourages collaboration and transparency by prompting open discussions among team members before and during project execution. It is particularly suited for projects where diverse expertise and perspectives are involved, such as those integration technical and social aspects of data science.
The Markkula Center Ethical Decision-Making Framework, on the other hand, provides a structured, step-by-step method for analyzing ethical dilemmas. It guides individuals through five stages: recognizing an ethical issue, gathering the facts, evaluating alternative actions using various ethical lenses (utilitarian, rights, justice, common good, care ethics and virtue), making a decision and reflecting on the outcome. This model emphasizes personal moral reasoning and ethical justification rather than collective brainstorming. It is often used in business, medical and educational contexts to support sound ethical judgement in complex decision-making situations.

2.	Similarities and differences
Both models are designed to improve ethical awareness and help decision-makers act responsibly. The models share a common goal of integrating ethics into professional practice and encourage reflection on potential consequences. However, their focus and structure differ considerably.
The Data Ethics Canvas is a collaborative, preventive and design-oriented model. The model helps teams anticipate and prevent ethical issues early in a project. The Markkula Framework on the other hand is a individual, procedural an decision-oriented model. The model focusses on analyzing dilemmas that have already emerged. The Canvas uses open-ended prompts to stimulate the group discussion, while the Markkula model relies on ethical theories and moral reasoning to weigh alternative actions.
Another difference between the models is the context of them. The Data Ethics Canvas was created specifically for data-driven and AI-related projects. The Markkula model on the other hand was designed for a more general purpose. This framework is applicable to various domains. The Canvas therefore addresses issues as data privacy, bias and transparency more directly. The Markkula framework offers a broader philosophical grounding.

3.	Personal preference
My personal preference goes out to the Data Ethics Canvas. This model has a collaborative and visual nature that makes it more practical for data science projects. The Canvas is made to be used in groups and encourages continuous ethical reflection. This makes it easier to integrate ethical thinking into projects.

4.	More suitable for data ethics (opinion)
The Data Ethics Canvas is in my opinion more suitable for data ethics. The model focuses on the data lifecycle and the stakeholder impact. This makes it relevant to challenges in AI and analytics (for example bias, consent and transparency). The model can be used by both technical and non-technical members. Also, the model encourages open ethical dialogue, which is of high value in a project workgroup. Al these points considered together makes me choose for the Canvas.
5.	References (APA)
Open Data Institute. (2017). The Data Ethics Canvas. The Open Data Institute. https://theodi.org/article/data-ethics-canvas/

Markkula Center for Applied Ethics. (2009). A framework for ethical decision making. Santa Clara University. https://www.scu.edu/ethics/ethics-resources/a-framework-for-ethical-decision-making/


Assignment 2:

First impression
Reading the article about the Breeze dating app, gave me a good insight in how Ethics is important in Data Science. algorithmic bias and discrimination can get into the algorithm without it being noticed at first. The algorithm Breeze used may have treated users unequally based on race or ethnicity. 
Due to lack of transparency in the model’s decision making, it’s hard to tackle this problem. If users are less frequently matched due to ethnic backgrounds, this could be seen as discrimination, and be in violation with the law.
Directed Acyclic Graph (DAG)
With the help of a DAG, the stream of the algorithm can be visualized. With this, it is possible to see how bias may have entered the system. Below here the DAG of this dilemma is made:








FOR THE DAG, SEE THE DAG.PNG











If the users behave based on ethnicity, the training data is also influenced by these choices. The algorithm can learn to replicate this bias. In this step, discrimination is added to the algorithm. This is something the app owners want to avoid.


Reflection after drawing DAG
Writing down the DAG, it made me clear that the user behavior can have a real big impact on how the algorithm is trained. Also the user demographic information play a big part in this. Even in this information, discrimination can be created. 
Also the feedback loop is a part that can be overlooked fast. If the algorithm picks up on the user behavior, it will adapt to it. Then, if the algorithm sees this is successful, it will keep doing this more and more. In time, the users will only see matches similar to their own background. The data reinforces the model’s belief that these matches are preferred.
To get rid of the unfairness, a lot needs to be done. Just removing some attributes like ethnicity will not do the trick. Demographic variables will indirectly lead to the same conclusion. So to get fairness correct in the algorithm, not only direct bias, but also indirect streams of bias need to be looked at.

Recommendations to data scientists
When data scientists get assigned a task like this, the following steps are crucial:
-	Preform tests on bias and fairness, by testing outcomes for demographic groups.
-	Use explainable techniques to make the decisions the models makes interpretable.
-	Make use of different experts, especially on terms of law and ethics. Keep them updated with the model design and evaluation.
-	Document everything well
-	Keep monitoring the model outcomes, so if unfairness gets into it, you can react quickly.
Short summary, keep everything transparent. Especially when you are working with algorithms that are build to help with human relationships, users need to be treated equally.














References

College voor de Rechten van de Mens. (2023, September 6). Datingapp Breeze mag en moet algoritme aanpassen om discriminatie te voorkomen. Retrieved from https://www.mensenrechten.nl/actueel/nieuws/2023/09/06/dating-app-breeze-mag-en-moet-algoritme-aanpassen-om-discriminatie-te-voorkomen
