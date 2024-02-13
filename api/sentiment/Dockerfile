
FROM public.ecr.aws/lambda/python:3.11
# Copy requirements.txt
RUN echo "scikit-learn==1.4.0" > ${LAMBDA_TASK_ROOT}/requirements.txt
RUN echo "tensorflow==2.15.0" >> ${LAMBDA_TASK_ROOT}/requirements.txt
RUN echo "gensim==4.3.2" >> ${LAMBDA_TASK_ROOT}/requirements.txt

# Install the specified packages
RUN pip install -r requirements.txt

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY artifacts ${LAMBDA_TASK_ROOT}/artifacts

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "lambda_function.handler" ]
