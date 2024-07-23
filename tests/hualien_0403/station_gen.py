# Open the input file for reading
with open('/home/patrick/Work/EQNet/tests/hualien_0403/station.txt', 'r') as infile:
    # Open the output file for writing
    with open('/home/patrick/Work/AutoQuake/Reloc2/Hualien_das.all.select', 'w') as outfile:
        # Iterate over each line in the input file
        for line in infile:
            # Split the line into fields
            fields = line.strip().split()
            sta = fields[0]
            lat = fields[1]
            lon = fields[2]
            dep = fields[3]
            if sta[:1] == '0':
                sta = f"A{sta[1:]}"
            elif sta[:1] == '1':
                sta = f"B{sta[1:]}"
            # Append the new columns
            new_line = f"{sta} {lat} {lon} {dep} 19010101 21001231\n"
            
            # Write the updated line to the output file
            outfile.write(new_line)
