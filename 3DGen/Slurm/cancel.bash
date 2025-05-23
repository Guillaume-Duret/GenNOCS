for job in $(squeue -u $USER -o "%.18i" | grep '_99'); do
    #scancel "$job"
    echo $job
done


echo "TEST"
# Get the list of job IDs for the user with the specified array format
squeue -u $USER -o "%.18i" | while read -r job_id; do
    # Extract the base job ID by removing the array part
    base_job_id=$(echo "$job_id" | cut -d'[' -f1)

    # Cancel the specific array element (index 99)
    scancel "${base_job_id}[0-96]"
    echo "${base_job_id}[0-96]"
done
