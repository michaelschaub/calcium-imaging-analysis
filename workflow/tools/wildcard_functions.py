
def subjects_from_wildcard(wildcard):
    '''
    infers the individual subjects and dates from the concatenated {subject_dates} wildcard
    depending on whether combine_session was set true this means loading indivdual sessions or loading all session together (and transform spatial components)
    '''
    single_sessions = wildcard.split("#")
    subject_dates = {}

    for session in single_sessions:
        subject_date_list = session.split(".")
        subjects = subject_date_list[0::3]
        dates = subject_date_list[1::3]
        
        for i,s in enumerate(subjects):
            subject_dates[s] =  subject_dates.get(s,[])
            subject_dates[s].append(dates[i])


    return subject_dates